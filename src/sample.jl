
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
                                  effective_lengths[t.metadata.id])
            if isfinite(fragpr) && fragpr > MIN_FRAG_PROB
                i_ = alnpr.metadata.mate1_idx > 0 ?
                        rs.alignments[alnpr.metadata.mate1_idx].id :
                        rs.alignments[alnpr.metadata.mate2_idx].id
                i = aln_idx_map[i_]

                thrid = Threads.threadid()
                push!(Is[thrid], i)
                push!(Js[thrid], t.metadata.id)
                push!(Vs[thrid], fragpr)
            end
        end
    end

    return (vcat(Is...), vcat(Js...), vcat(Vs...))
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

    @time ts, ts_metadata = Transcripts(transcripts_filename, excluded_transcripts)
    @time rs = Reads(reads_filename, excluded_seqs)
    read_transcript_sequences!(ts, genome_filename)
    fm = FragModel(rs, ts)

    println("intersecting reads and transcripts...")

    # reassign indexes to alignments to group by position
    aln_idx_map = zeros(Int, length(rs.alignments))
    nextidx = 1
    # for alnpr in rs.alignment_pairs
    # for alnpr in rs.alignment_pairs
    for tree in values(rs.alignment_pairs.trees)
        for alnpr in tree
            if alnpr.metadata.mate1_idx > 0
                id = rs.alignments[alnpr.metadata.mate1_idx].id
                if aln_idx_map[id] == 0
                    aln_idx_map[id] = nextidx
                    nextidx += 1
                end
            else
                id = rs.alignments[alnpr.metadata.mate2_idx].id
                if aln_idx_map[id] == 0
                    aln_idx_map[id] = nextidx
                    nextidx += 1
                end
            end
        end
    end

    tic()

    effective_lengths = Float32[effective_length(fm, t) for t in ts]
    I, J, V = parallel_intersection_loop(ts, rs, fm, effective_lengths, aln_idx_map)

    toc()

    # reassign indexes to remove any zero rows, which would lead to a
    # -Inf log likelihood
    p = sortperm(I)
    I = I[p]
    J = J[p]
    V = V[p]

    if isempty(I)
        warn("No compatible reads found.")
        m = 0
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
        m = maximum(I)
    end

    n = length(ts)
    M = sparse(I, J, V, m, n)

    if !isnull(output)
        h5open(get(output), "w") do out
            out["m"] = M.m
            out["n"] = M.n
            out["colptr", "compress", 1] = M.colptr
            out["rowval", "compress", 1] = M.rowval
            out["nzval", "compress", 1] = M.nzval
            out["effective_lengths", "compress", 1] = effective_lengths
            g = g_create(out, "metadata")
            attrs(g)["gfffilename"] = ts_metadata.filename
            attrs(g)["gffhash"]     = ts_metadata.gffhash
            attrs(g)["gffsize"]     = ts_metadata.gffsize
            attrs(g)["excluded_transcripts_hash"]     = ts_metadata.excluded_transcripts_hash
        end
    end

    return RNASeqSample(m, n, M, effective_lengths, ts_metadata)
end


