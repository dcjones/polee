
type RNASeqSample
    m::Int
    n::Int
    X::SparseMatrixRSB
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
    X = SparseMatrixRSB(SparseMatrixCSC(m, n, colptr, rowval, nzval))
    effective_lengths = read(input["effective_lengths"])

    return RNASeqSample(m, n, X, effective_lengths, TranscriptsMetadata())
end


function parallel_intersection_loop(ts, rs, fm, effective_lengths, aln_idx_map, I, J, V)
    # join matching trees from ts and rs
    T = Tuple{GenomicFeatures.ICTree{TranscriptMetadata},
              GenomicFeatures.ICTree{AlignmentPairMetadata}}
    treepairs = Array{T}(0)
    for (seqname, ts_tree) in ts.trees
        if haskey(rs.alignment_pairs.trees, seqname)
            push!(treepairs, (ts_tree, rs.alignment_pairs.trees[seqname]))
        end
    end

    # process pairs of trees in parallel
    mut = Threads.Mutex()
    Threads.@threads for treepair_idx in 1:length(treepairs)
        ts_tree, rs_tree = treepairs[treepair_idx]
        for (t, alnpr) in intersect(ts_tree, rs_tree)
            fragpr = condfragprob(fm, t, rs, alnpr,
                                  effective_lengths[t.metadata.id])
            if isfinite(fragpr) && fragpr > MIN_FRAG_PROB
                i_ = alnpr.metadata.mate1_idx > 0 ?
                        rs.alignments[alnpr.metadata.mate1_idx].id :
                        rs.alignments[alnpr.metadata.mate2_idx].id
                i = aln_idx_map[i_]

                lock(mut)
                push!(I, i)
                push!(J, t.metadata.id)
                push!(V, fragpr)
                unlock(mut)
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
                      output=Nullable{String}())

    ts, ts_metadata = Transcripts(transcripts_filename)
    rs = Reads(reads_filename, excluded_seqs)
    println("reading transcript sequences")
    read_transcript_sequences!(ts, genome_filename)
    println("done")
    fm = FragModel(rs, ts)

    println("intersecting reads and transcripts...")

    # sparse matrix indexes and values
    I = UInt32[]
    J = UInt32[]
    V = Float32[]

    # reassign indexes to alignments to group by position
    aln_idx_map = zeros(Int, length(rs.alignments))
    nextidx = 1
    for alnpr in rs.alignment_pairs
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

    tic()

    effective_lengths = Float32[effective_length(fm, t) for t in ts]
    parallel_intersection_loop(ts, rs, fm, effective_lengths, aln_idx_map, I, J, V)

    toc()

    # reassign indexes to remove any zero rows, which would lead to a
    # -Inf log likelihood
    p = sortperm(I)
    I = I[p]
    J = J[p]
    V = V[p]
    next_i = 1
    for k in 2:length(I)
        I[k] = min(I[k], I[k-1] + 1)
    end

    m = maximum(I)
    n = length(ts)

    if !isnull(output)
        M = sparse(I, J, V, m, n)
        h5open(get(output), "w") do out
            out["m"] = M.m
            out["n"] = M.n
            out["colptr", "compress", 1] = M.colptr
            out["rowval", "compress", 1] = M.rowval
            out["nzval", "compress", 1] = M.nzval
            out["effective_lengths", "compress", 1] = effective_lengths
            # g = g_create(out, "metadata")
            # attrs(g)["gfffilename"] = metadata.gfffilename
            # attrs(g)["gffhash"]     = metadata.gffhash
            # attrs(g)["gffsize"]     = metadata.gffsize
        end
    end

    return RNASeqSample(m, n, SparseMatrixRSB(I, J, V, m, n),
                        effective_lengths, ts_metadata)
end


