
"""
Representation of a an RNA-Seq with a sparse read-transcript compatibility
matrix.
"""
mutable struct RNASeqSample
    m::Int # number of reads
    n::Int # number of transcripts

    # X[i,j] gives the probability of observing read i from transcript j.
    X::SparseMatrixCSC{Float32, UInt32}

    # effective_lengths[j] if the effective length of transcript j
    effective_lengths::Vector{Float32}

    # transcript set and accompanying metadata
    ts::Transcripts
    transcript_metadata::TranscriptsMetadata

    # sequences filename
    sequences_filename::String
    sequences_file_hash::Vector{UInt8}
end


function Base.size(s::RNASeqSample)
    return (s.m, s.n)
end


"""
Read a likelihood matrix from a h5 file.
"""
function Base.read(filename::String, ::Type{RNASeqSample})
    input = h5open(filename, "r")
    m = read(input["m"]) # number of reads
    n = read(input["n"]) # number of transcripts
    colptr = read(input["colptr"])
    rowval = read(input["rowval"])
    nzval = read(input["nzval"])
    X = SparseMatrixCSC(m, n, colptr, rowval, nzval)
    effective_lengths = read(input["effective_lengths"])

    return RNASeqSample(m, n, X, effective_lengths, Transcripts(), TranscriptsMetadata())
end


"""
Intersect transcripts and reads very efficiently.
  * ts: transcripts
  * rs: reads
  * fm: fragment model
  * effective_lengths: effective lengths
  * aln_idx_map: map read indexes to indexs in the compatibility matrix
"""
function parallel_intersection_loop( ts, rs, fm, effective_lengths, aln_idx_map)
    # join matching trees from ts and rs
    T = Tuple{GenomicFeatures.ICTree{TranscriptMetadata},
              GenomicFeatures.ICTree{AlignmentPairMetadata}}
    treepairs = T[]
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


"""
Multithreaded core loop for intersecting transcripts and read alignments.

(Julia sometimes has problems with multithreaded loops if it isn't basically
in its own function like this.)
"""
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
    I = vcat(Is...)
    J = vcat(Js...)
    V = vcat(Vs...)
    return (I, J, V)
end


# reassign indexes to remove any zero rows, which would lead to a
# -Inf log likelihood
function compact_indexes!(I, aln_idx_rev_map::Vector{UInt32})
    if isempty(I)
        @warn "No compatible reads found."
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
                      reads_filename::Union{IO, String},
                      excluded_seqs::Set{String},
                      excluded_transcripts::Set{String},
                      output=Nullable{String}();
                      no_bias::Bool=false,
                      dump_bias_training_examples::Bool=false)
    ts, ts_metadata = Transcripts(transcripts_filename, excluded_transcripts)
    @tic()
    read_transcript_sequences!(ts, genome_filename)
    sequences_file_hash = SHA.sha1(open(genome_filename))
    @toc("Reading transcripts")
    return RNASeqSample(
        ts, ts_metadata, reads_filename, excluded_seqs,
        excluded_transcripts, genome_filename,
        sequences_file_hash, output, no_bias=no_bias,
        dump_bias_training_examples=dump_bias_training_examples)
end


"""
Build a Transcripts set assuming each entry in a fasta file is one transcript.
"""
function read_transcripts_from_fasta(filename, excluded_transcripts)
    @tic()
    reader = open(FASTA.Reader, filename)
    entry = eltype(reader)()

    transcripts = Transcript[]
    while !isnull(tryread!(reader, entry))
        seqname = FASTA.identifier(entry)
        if seqname ∈ excluded_transcripts
            continue
        end
        seq = FASTA.sequence(entry)
        id = length(transcripts) + 1
        t = Transcript(seqname, 1, length(seq), STRAND_POS,
                TranscriptMetadata(seqname, id, [Exon(1, length(seq))], seq))
        push!(transcripts, t)
    end
    close(reader)

    ts = Transcripts(transcripts, true)
    ts_metadata = TranscriptsMetadata()

    @toc("Reading transcript sequences")
    println("Read ", length(transcripts), " transcripts")

    return ts, ts_metadata
end


"""
Build an RNASeqSample from scratch where transcript sequences and alignments
are given instead of genome sequence and alginments.
"""
function RNASeqSample(transcript_sequence_filename::String,
                      reads_filename::Union{IO, String},
                      excluded_seqs::Set{String},
                      excluded_transcripts::Set{String},
                      output=Nullable{String}();
                      no_bias::Bool=false,
                      dump_bias_training_examples::Bool=false)

    ts, ts_metadata = read_transcripts_from_fasta(
        transcript_sequence_filename, excluded_transcripts)
    sequences_file_hash = SHA.sha1(open(transcript_sequence_filename))

    return RNASeqSample(
        ts, ts_metadata, reads_filename, excluded_seqs,
        excluded_transcripts, transcript_sequence_filename,
        sequences_file_hash, output, no_bias=no_bias,
        dump_bias_training_examples=dump_bias_training_examples)
end


"""
Build an RNASeqSample from Transcripts, with sequences with the metadata.seq
field set.
"""
function RNASeqSample(ts::Transcripts,
                      ts_metadata::TranscriptsMetadata,
                      reads_filename::Union{IO, String},
                      excluded_seqs::Set{String},
                      excluded_transcripts::Set{String},
                      sequences_filename::String,
                      sequences_file_hash::Vector{UInt8},
                      output=Nullable{String}();
                      no_bias::Bool=false,
                      num_training_reads::Int=200000,
                      dump_bias_training_examples::Bool=false)

    @tic()
    rs = Reads(reads_filename, excluded_seqs)
    @toc("Reading BAM file")

    # train fragment model by selecting a random subset of reads, assigning
    # them to transcripts, then extracting sequence context.
    rs_train      = subsample_reads(rs, num_training_reads)
    simplistic_fm = SimplisticFragModel(rs_train, ts)

    if !no_bias
        aln_idx_rev_map_ref = Ref{Vector{UInt32}}()
        sample_train  = RNASeqSample(
            simplistic_fm, rs_train, ts, ts_metadata,
            "", UInt8[], Nullable{String}(), Nullable(aln_idx_rev_map_ref))
        aln_idx_rev_map = aln_idx_rev_map_ref.x

        # assign reads to transcripts
        xs_train = optimize_likelihood(sample_train)
        Xt = SparseMatrixCSC(transpose(sample_train.X))

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

        fm = BiasedFragModel(
            rs_train, ts, read_assignments, dump_bias_training_examples)
    else
        fm = simplistic_fm
    end

    return RNASeqSample(
        fm, rs, ts, ts_metadata, sequences_filename, sequences_file_hash, output)
end


"""
Build a RNASeqSample from a provided fragment model and read set.
"""
function RNASeqSample(fm::FragModel,
                      rs::Reads,
                      ts::Transcripts,
                      ts_metadata::TranscriptsMetadata,
                      sequences_filename::String,
                      sequences_file_hash::Vector{UInt8},
                      output=Nullable{String}(),
                      aln_idx_rev_map_ref=Nullable{Ref{Vector{UInt32}}}())

    # reassign indexes to alignments to group by position
    # TODO: this is allocated larger than it needs to be. Use a dict?
    aln_idx_map = zeros(UInt32, length(rs.alignments))
    nextidx = 1
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

    println("computing effective lengths")
    effective_lengths = Vector{Float32}(undef, length(ts))
    ts_arr = collect(ts)

    # 71 seconds
    Threads.@threads for t in ts_arr
        effective_lengths[t.metadata.id] = effective_length(fm, t)
    end

    # if isa(fm, BiasedFragModel)
    #     open("effective-lengths.csv", "w") do output
    #         println(output, "transcript_id,efflen")
    #         for t in ts
    #             println(output, t.metadata.name, ",", effective_lengths[t.metadata.id])
    #         end
    #     end
    #     exit()
    # end

    println("intersecting reads and transcripts...")

    # open("effective-lengths.csv", "w") do out
    #     println(out, "transcript_id,tlen,efflen")
    #     for t in ts
    #         println(
    #             out,
    #             t.metadata.name, ",",
    #             length(t.metadata.seq), ",",
    #             effective_lengths[t.metadata.id])
    #     end
    # end

    @tic()
    I, J, V = parallel_intersection_loop(ts, rs, fm, effective_lengths, aln_idx_map)
    @toc("Intersecting reads and transcripts")

    # reverse index (mapping matrix index to read id)
    @tic()
    aln_idx_rev_map = zeros(UInt32, maximum(aln_idx_map))
    for (i, j) in enumerate(aln_idx_map)
        if j != 0
            @assert aln_idx_rev_map[j] == 0
            aln_idx_rev_map[j] = i
        end
    end
    @assert sum(aln_idx_rev_map .== 0) == 0

    GC.gc()
    p = sortperm(I)
    I = I[p]
    J = J[p]
    V = V[p]
    GC.gc()

    aln_idx_rev_map = compact_indexes!(I, aln_idx_rev_map)
    if !isnull(aln_idx_rev_map_ref)
        get(aln_idx_rev_map_ref).x = aln_idx_rev_map
    end
    @toc("Rearranging sparse matrix indexes")

    m = isempty(I) ? 0 : Int(maximum(I))
    n = length(ts)
    M = sparse(I, J, V, m, n)

    println(length(V), " intersections")
    @printf("%0.1f%% reads accounted for (compatible with at least one transcript)\n",
        100.0 * (m / rs.num_reads))

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

    return RNASeqSample(
        m, n, M, effective_lengths, ts, ts_metadata,
        sequences_filename, sequences_file_hash)
end

