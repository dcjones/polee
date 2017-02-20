
type RNASeqSample
    m::Int
    n::Int
    X::RSBMatrix
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

    return RNASeqSample(m, n, X)
end


"""
Build an RNASeqSample from scratch.
"""
function RNASeqSample(transcripts_filename::String,
                      genome_filename::String,
                      reads_filename::String;
                      output=Nullable{String}())

    rs = Reads(reads_filename)
    ts = Transcripts(transcripts_filename)
    read_transcript_sequences!(ts, genome_filename)
    fm = FragModel(rs, ts)

    println("intersecting reads and transcripts...")

    # sparse matrix indexes and values
    I = Int32[]
    J = Int32[]
    V = Float32[]
    intersection_count = 0
    intersection_candidate_count = 0

    MIN_FRAG_PROB = 1e-8

    # reassign indexes to alignments to group by position
    aln_idx_map = Dict{Int, Int}()
    for alnpr in rs.alignment_pairs
        if alnpr.metadata.mate1_idx > 0
            get!(aln_idx_map, rs.alignments[alnpr.metadata.mate1_idx].id,
                 length(aln_idx_map) + 1)
        else
            get!(aln_idx_map, rs.alignments[alnpr.metadata.mate2_idx].id,
                 length(aln_idx_map) + 1)
        end
    end

    # reassign transcript indexes to group by position
    for (tid, t) in enumerate(ts)
        t.metadata.id = tid
    end

    tic()
    for (t, alnpr) in intersect(ts, rs.alignment_pairs)
        intersection_candidate_count += 1
        fragpr = condfragprob(fm, t, rs, alnpr)
        if fragpr > MIN_FRAG_PROB
            i = alnpr.metadata.mate1_idx > 0 ?
                    rs.alignments[alnpr.metadata.mate1_idx].id :
                    rs.alignments[alnpr.metadata.mate2_idx].id
            push!(I, aln_idx_map[i])
            #push!(J, t.metadata.id + 1) # +1 to make room for pseudotranscript
            push!(J, t.metadata.id)
            push!(V, fragpr)
        end
    end

    # Write transcript out with corresponding indexes
    open("transcripts.txt", "w") do out
        for t in ts
            println(out, t.metadata.id, ",", t.metadata.name)
        end
    end

    # conditional probability of observing a fragment given it belongs to some
    # other unknown transcript or something else. TODO: come up with some
    # principled number of this.
    #const SPURIOSITY_PROB = MIN_FRAG_PROB
    #for i in 1:maximum(I)
        #push!(I, i)
        #push!(J, 1)
        #push!(V, SPURIOSITY_PROB)
    #end
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
        end
    end

    return RNASeqSample(m, n, RSBMatrix(I, J, V, m, n))
end


