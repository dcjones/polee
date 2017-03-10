
type RNASeqSample
    m::Int
    n::Int
    X::RSBMatrix
    effective_lengths::Vector{Float32}
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
    X = RSBMatrix(SparseMatrixCSC(m, n, colptr, rowval, nzval))
    effective_lengths = read(input["effective_lengths"])

    return RNASeqSample(m, n, X, effective_lengths)
end


function parallel_intersection_loop(ts, rs, fm, aln_idx_map, I, J, V)
    MIN_FRAG_PROB = 1e-10

    # join matching trees from ts and rs
    T = Tuple{Intervals.IntervalCollectionTree{TranscriptMetadata},
              Intervals.IntervalCollectionTree{AlignmentPairMetadata}}
    treepairs = Array(T, 0)
    for (seqname, ts_tree) in ts.trees
        if haskey(rs.alignment_pairs.trees, seqname)
            push!(treepairs, (ts_tree, rs.alignment_pairs.trees[seqname]))
        end
    end

    @show length(treepairs)

    # process pairs of trees in parallel
    mut = Threads.Mutex()
    Threads.@threads for treepair_idx in 1:length(treepairs)
        ts_tree, rs_tree = treepairs[treepair_idx]
        for (t, alnpr) in intersect(ts_tree, rs_tree)
            fragpr = condfragprob(fm, t, rs, alnpr)
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

    ts = Transcripts(transcripts_filename)
    rs = Reads(reads_filename, excluded_seqs)
    read_transcript_sequences!(ts, genome_filename)
    fm = FragModel(rs, ts)

    println("intersecting reads and transcripts...")

    # sparse matrix indexes and values
    I = Int32[]
    J = Int32[]
    V = Float32[]

    # reassign indexes to alignments to group by position
    tic()
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

    # reassign transcript indexes to group by position
    for (tid, t) in enumerate(ts)
        t.metadata.id = tid
    end
    toc()

    tic()
    parallel_intersection_loop(ts, rs, fm, aln_idx_map, I, J, V)

    effective_lengths = Float32[effective_length(fm, t) for t in ts]

    # Write transcript out with corresponding indexes
    # TODO: this should probably be dumped into the output somehow
    open("transcripts.txt", "w") do out
        for t in ts
            println(out, t.metadata.id, ",", t.metadata.name)
        end
    end

    toc()

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
        end
    end

    return RNASeqSample(m, n, RSBMatrix(I, J, V, m, n), effective_lengths)
end


