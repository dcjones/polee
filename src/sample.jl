
type RNASeqSample
    m::Int
    n::Int
    X::SparseMatrixCSC{Float32, UInt32}
    effective_lengths::Vector{Float32}
    transcript_metadata::TranscriptsMetadata
end


function Base.size(s::RNASeqSample)
    return (s.m, s.n)
end


function Base.read(filename::String, ::Type{RNASeqSample})
    input = h5open(filename, "r")
    m = read(input["m"]) # number of reads
    n = read(input["n"]) # number of transcripts
    colptr = read(input["colptr"])
    rowval = read(input["rowval"])
    nzval = read(input["nzval"])
    X = SparseMatrixCSC(m, n, colptr, rowval, nzval)
    effective_lengths = read(input["effective_lengths"])

    return RNASeqSample(m, n, X, effective_lengths, TranscriptsMetadata())
end


function parallel_intersection_loop(ts, rs, fm, effective_lengths, aln_idx_map)
    # join matching trees from ts and rs
    T = Tuple{GenomicFeatures.ICTree{TranscriptMetadata},
              GenomicFeatures.ICTree{AlignmentPairMetadata}}
    treepairs = Array{T}(0)
    for (seqname, ts_tree) in ts.trees
        if haskey(rs.alignment_pairs.trees, seqname)
            push!(treepairs, (ts_tree, rs.alignment_pairs.trees[seqname]))
        end
    end

    return parallel_intersection_loop_inner(treepairs, rs, fm, effective_lengths, aln_idx_map)
end


"""
True if b is contained in a.
"""
function intersect_contains(a, b)
    return a.first <= b.first && b.last <= a.last
end


function parallel_intersection_loop_inner(treepairs, rs, fm, effective_lengths, aln_idx_map)
    Is = [UInt32[] for _ in 1:Threads.nthreads()]
    Js = [UInt32[] for _ in 1:Threads.nthreads()]
    Vs = [Float32[] for _ in 1:Threads.nthreads()]

    Threads.@threads for treepair_idx in 1:length(treepairs)
    # for treepair_idx in 1:length(treepairs)
        ts_tree, rs_tree = treepairs[treepair_idx]
        for (t, alnpr) in intersect(ts_tree, rs_tree, intersect_contains)
            fragpr = condfragprob(fm, t, rs, alnpr,
                                  effective_lengths[Int(t.metadata.id)])
            if isfinite(fragpr) && fragpr > MIN_FRAG_PROB
                i_ = alnpr.metadata.mate1_idx > 0 ?
                        rs.alignments[alnpr.metadata.mate1_idx].id :
                        rs.alignments[alnpr.metadata.mate2_idx].id
                i = aln_idx_map[Int(i_)]

                thrid = Threads.threadid()
                push!(Is[thrid], i)
                push!(Js[thrid], t.metadata.id)
                push!(Vs[thrid], fragpr)
            end
        end
    end

    N = 0
    for Ii in Is
        N += length(Ii)
    end

    I = Array{UInt32}(N)
    J = Array{UInt32}(N)
    V = Array{Float32}(N)
    j = 1
    for i in 1:length(Is)
        copy!(I, j, Is[i])
        copy!(J, j, Js[i])
        copy!(V, j, Vs[i])
        j += length(Is[i])
    end

    return (I, J, V)
end


# reassign indexes to remove any zero rows, which would lead to a
# -Inf log likelihood
function compact_indexes!(I)
    if isempty(I)
        warn("No compatible reads found.")
    else
        last_i = I[1]
        I[1] = 1
        for k in 2:length(I)
            if I[k] == last_i
                I[k] = I[k-1]
            else
                last_i = I[k]
                I[k] = I[k-1] + 1
            end
        end
    end
end



"""
Build an RNASeqSample from scratch.
"""
function RNASeqSample(transcripts_filename::String,
                      genome_filename::String,
                      reads_filename::String,
                      excluded_seqs::Set{String},
                      excluded_transcripts::Set{String},
                      output=Nullable{String}())

    ts, ts_metadata = Transcripts(transcripts_filename, excluded_transcripts)
    read_transcript_sequences!(ts, genome_filename)
    rs = Reads(reads_filename, excluded_seqs)
    fm = FragModel(rs, ts)

    println("intersecting reads and transcripts...")

    # reassign indexes to alignments to group by position
    aln_idx_map = zeros(UInt32, length(rs.alignments))
    nextidx = 1
    # for alnpr in rs.alignment_pairs
    # for alnpr in rs.alignment_pairs
    for tree in values(rs.alignment_pairs.trees)
        for alnpr in tree
            if alnpr.metadata.mate1_idx > 0
                id = Int(rs.alignments[alnpr.metadata.mate1_idx].id)
                if aln_idx_map[id] == 0
                    aln_idx_map[id] = nextidx
                    nextidx += 1
                end
            else
                id = Int(rs.alignments[alnpr.metadata.mate2_idx].id)
                if aln_idx_map[id] == 0
                    aln_idx_map[id] = nextidx
                    nextidx += 1
                end
            end
        end
    end

    effective_lengths = Float32[effective_length(fm, t) for t in ts]
    @time I, J, V = parallel_intersection_loop(ts, rs, fm, effective_lengths, aln_idx_map) # 2.829 GB (53% GC)

    gc()
    p = sortperm(I)
    I = I[p]
    J = J[p]
    V = V[p]
    gc()

    compact_indexes!(I) # 0.000 GB
    m = isempty(I) ? 0 : Int(maximum(I))

    n = length(ts)
    M = sparse(I, J, V, m, n)

    # @show sum(J .== 133568)
    # @show sum(J .== 133569)
    # @show sum(J .== 133570)

    # @show effective_lengths[133568]
    # @show effective_lengths[133569]
    # @show effective_lengths[133570]

    @show extrema(V[J .== 133568])
    @show extrema(V[J .== 133569])
    @show extrema(V[J .== 133570])

    # TODO: decide what to do about this
    if !isnull(output)
        warn("Outputing likelihood matrix currently disabled.")
    #     h5open(get(output), "w") do out
    #         out["m"] = M.m
    #         out["n"] = M.n
    #         out["colptr", "compress", 1] = M.colptr
    #         out["rowval", "compress", 1] = M.rowval
    #         out["nzval", "compress", 1] = M.nzval
    #         out["effective_lengths", "compress", 1] = effective_lengths
    #         g = g_create(out, "metadata")
    #         attrs(g)["gfffilename"] = ts_metadata.filename
    #         attrs(g)["gffhash"]     = ts_metadata.gffhash
    #         attrs(g)["gffsize"]     = ts_metadata.gffsize
    #         attrs(g)["excluded_transcripts_hash"]     = ts_metadata.excluded_transcripts_hash
    #     end
    end

    return RNASeqSample(m, n, M, effective_lengths, ts_metadata)
end


