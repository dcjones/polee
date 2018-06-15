
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
function compact_indexes!(I, aln_idx_rev_map::Vector{UInt32})
    if isempty(I)
        warn("No compatible reads found.")
    else
        numrows = 1
        last_i = I[1]
        for k in 2:length(I)
            if I[k] == last_i
            else
                numrows += 1
                last_i = I[k]
            end
        end

        aln_idx_rev_map2 = zeros(UInt32, numrows)
        last_i = I[1]
        aln_idx_rev_map2[1] = aln_idx_rev_map[I[1]]
        I[1] = 1
        for k in 2:length(I)
            if I[k] == last_i
                I[k] = I[k-1]
            else
                last_i = I[k]
                next_i = I[k-1] + 1
                aln_idx_rev_map2[next_i] = aln_idx_rev_map[I[k]]
                I[k] = next_i
            end
        end
    end

    return aln_idx_rev_map2
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
    return RNASeqSample(
        ts, ts_metadata, reads_filename, excluded_seqs,
        excluded_transcripts, output)
end


"""
Build an RNASeqSample from scratch where transcript sequences and alignments
are given instead of genome sequence and alginments.
"""
function RNASeqSample(transcript_sequence_filename::String,
                      reads_filename::String,
                      excluded_seqs::Set{String},
                      excluded_transcripts::Set{String},
                      output=Nullable{String}())

    println("Reading transcript sequences")
    tic()
    reader = open(FASTA.Reader, transcript_sequence_filename)
    entry = eltype(reader)()

    transcripts = Transcript[]
    while !isnull(tryread!(reader, entry))
        seqname = FASTA.identifier(entry)
        seq = FASTA.sequence(entry)
        id = length(transcripts) + 1
        t = Transcript(seqname, 1, length(seq), STRAND_POS,
                TranscriptMetadata(seqname, id, [Exon(1, length(seq))], seq))
        push!(transcripts, t)
    end

    ts = Transcripts(transcripts, true)
    ts_metadata = TranscriptsMetadata()

    toc()
    println("Read ", length(transcripts), " transcripts")

    return RNASeqSample(
        ts, ts_metadata, reads_filename, excluded_seqs,
        excluded_transcripts, output)
end


"""
Build an RNASeqSample from Transcripts, with sequences with the metadata.seq
field set.
"""
function RNASeqSample(ts::Transcripts,
                      ts_metadata::TranscriptsMetadata,
                      reads_filename::String,
                      excluded_seqs::Set{String},
                      excluded_transcripts::Set{String},
                      output=Nullable{String}();
                      num_training_reads::Int=100000)

    rs = Reads(reads_filename, excluded_seqs)

    # train fragment model by selecting a random subset of reads, assigning
    # them to transcripts, then extracting sequence context.
    rs_train      = subsample_reads(rs, num_training_reads)
    simplistic_fm = SimplisticFragModel(rs_train, ts)
    aln_idx_rev_map_ref = Ref{Vector{UInt32}}()
    sample_train  = RNASeqSample(
        simplistic_fm, rs_train, ts, ts_metadata,
        Nullable{String}(), Nullable(aln_idx_rev_map_ref))
    aln_idx_rev_map = aln_idx_rev_map_ref.x

    # assign reads to transcripts
    xs_train = optimize_likelihood(sample_train)
    Xt = transpose(sample_train.X)

    # map read index to a transcript index
    read_assignments = Dict{Int, Int}()

    for i in 1:size(sample_train.X, 1)
        read_idx = aln_idx_rev_map[i]

        z_sum = 0.0
        for ptr in Xt.colptr[i]:Xt.colptr[i+1]-1
            j = Xt.rowval[ptr]
            z_sum += xs_train[j] * Xt.nzval[ptr]
        end

        # assign read randomly according to its probability
        r = rand()
        read_assignment = 0
        for ptr in Xt.colptr[i]:Xt.colptr[i+1]-1
            j = Xt.rowval[ptr]
            zj = xs_train[j] * Xt.nzval[ptr] / z_sum
            if zj > r || ptr == Xt.colptr[i+1]-1
                read_assignment = j
            else
                r -= zj
            end
        end
        @assert read_assignment != 0
        @assert !haskey(read_assignments, read_idx)
        read_assignments[read_idx] = read_assignment
    end

    fm = BiasedFragModel(rs_train, ts, read_assignments)

    return RNASeqSample(fm, rs, ts, ts_metadata, output)
end


"""
Build a RNASeqSample from a provided fragment model and read set.
"""
function RNASeqSample(fm::FragModel,
                      rs::Reads,
                      ts::Transcripts,
                      ts_metadata::TranscriptsMetadata,
                      output=Nullable{String}(),
                      aln_idx_rev_map_ref=Nullable{Ref{Vector{UInt32}}})

    println("intersecting reads and transcripts...")

    # reassign indexes to alignments to group by position
    # TODO: this is allocated larger than it needs to be. Use a dict?
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

    compute_transcript_bias!(fm, ts)
    effective_lengths = Float32[effective_length(fm, t) for t in ts]
    I, J, V = parallel_intersection_loop(ts, rs, fm, effective_lengths, aln_idx_map) # 2.829 GB (53% GC)

    # reverse index (mapping matrix index to read id)
    aln_idx_rev_map = zeros(UInt32, maximum(I))
    for (i, j) in enumerate(aln_idx_map)
        if j != 0
            @assert aln_idx_rev_map[j] == 0
            aln_idx_rev_map[j] = i
        end
    end
    @assert sum(aln_idx_rev_map .== 0) == 0

    gc()
    p = sortperm(I)
    I = I[p]
    J = J[p]
    V = V[p]
    gc()

    @show maximum(I)

    aln_idx_rev_map = compact_indexes!(I, aln_idx_rev_map)
    if !isnull(aln_idx_rev_map_ref)
        get(aln_idx_rev_map_ref).x = aln_idx_rev_map
    end

    m = isempty(I) ? 0 : Int(maximum(I))
    @show m

    n = length(ts)
    M = sparse(I, J, V, m, n)

    println(length(V), " intersections")

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