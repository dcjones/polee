
# A little experiment with building a model of the splicing code.

polee_dir = joinpath(dirname(@__FILE__), "..")

import Pkg
Pkg.activate(polee_dir)

using PyCall
pushfirst!(PyVector(pyimport("sys")["path"]), dirname(@__FILE__))
@pyimport splice_code as splice_code_py

import Polee
import YAML
using BioSequences
using GenomicFeatures
using Statistics
using HDF5


# K-mer size
const K = 6


function approximate_splicing_likelihood(cassette_exons, loaded_samples)
    feature_idxs = Int32[]
    feature_transcript_idxs = Int32[]
    antifeature_idxs = Int32[]
    antifeature_transcript_idxs = Int32[]
    num_features = length(cassette_exons)
    num_samples, n = size(loaded_samples.x0_values)

    feature_id = 0
    for (intron, flanks) in cassette_exons
        feature_id += 1
        @assert !isempty(intron.metadata)
        @assert !isempty(flanks.metadata[3])

        for id in flanks.metadata[3]
            push!(feature_idxs, feature_id)
            push!(feature_transcript_idxs, id)
        end

        for id in intron.metadata
            push!(antifeature_idxs, feature_id)
            push!(antifeature_transcript_idxs, id)
        end
    end

    feature_indices = hcat(feature_idxs .- 1, feature_transcript_idxs .- 1)
    antifeature_indices = hcat(antifeature_idxs .- 1, antifeature_transcript_idxs .- 1)

    sess = Polee.tf[:Session]()

    qx_feature_loc, qx_feature_scale = Polee.polee_py[:approximate_splicing_likelihood](
        loaded_samples.init_feed_dict, loaded_samples.variables,
        num_samples, num_features, n, feature_indices, antifeature_indices, sess)

    Polee.tf[:reset_default_graph]() # free up some memory
    sess[:close]()

    return (qx_feature_loc, qx_feature_scale)
end


function extract_sequence(seq, first, last)
    # TODO: may have to do some other stuff.
    return seq[first:last]
end



function one_hot_encode_seqs(seqs)
    arr = zeros(Float32, (length(seqs), length(seqs[1]), 4))
    for (i, seq) in enumerate(seqs)
        for (j, c) in enumerate(seq)
            k = ifelse(c == DNA_A, 1,
                ifelse(c == DNA_C, 2,
                ifelse(c == DNA_G, 3,
                ifelse(c == DNA_T, 4, 5))))
            if k < 5
                arr[i, j, k] = 1
            end
        end
    end

    return arr
end


function extract_sequence_features(genome_filename, conservations, cassette_exons)

    num_features = length(cassette_exons)

    donor_seqs        = Array{DNASequence}(undef, num_features)
    acceptor_seqs     = Array{DNASequence}(undef, num_features)
    alt_donor_seqs    = Array{DNASequence}(undef, num_features)
    alt_acceptor_seqs = Array{DNASequence}(undef, num_features)

    exonic_len = 10
    intronic_len = 30
    seqlen = exonic_len + intronic_len

    donor_cons        = zeros(Float32, (num_features, seqlen))
    acceptor_cons     = zeros(Float32, (num_features, seqlen))
    alt_donor_cons    = zeros(Float32, (num_features, seqlen))
    alt_acceptor_cons = zeros(Float32, (num_features, seqlen))

    reader = open(FASTA.Reader, genome_filename)
    entry = eltype(reader)()
    while Polee.tryread!(reader, entry)
        seqname = FASTA.identifier(entry)
        entryseq = FASTA.sequence(entry)

        if !haskey(conservations, seqname)
            conservation = zeros(Float32, length(entryseq))
        else
            conservation = conservations[seqname]
        end

        for (i, (intron, flanks)) in enumerate(cassette_exons)
            if intron.seqname != seqname
                continue
            end

            exon_first = flanks.metadata[1]
            exon_last  = flanks.metadata[2]

            donor_seq = extract_sequence(
                entryseq, intron.first-exonic_len, intron.first+intronic_len-1)
            acceptor_seq = extract_sequence(
                entryseq, intron.last-intronic_len+1, intron.last+exonic_len)
            alt_donor_seq = extract_sequence(
                entryseq, exon_last-exonic_len+1, exon_last+intronic_len)
            alt_acceptor_seq = extract_sequence(
                entryseq, exon_first-intronic_len, exon_first+exonic_len-1)

            donor_cons_i = conservation[intron.first-exonic_len:intron.first+intronic_len-1]
            acceptor_cons_i = conservation[intron.last-intronic_len+1:intron.last+exonic_len]
            alt_donor_cons_i = conservation[exon_last-exonic_len+1:exon_last+intronic_len]
            alt_acceptor_cons_i = conservation[exon_first-intronic_len:exon_first+exonic_len-1]

            if intron.strand == STRAND_NEG
                donor_seq, acceptor_seq, alt_donor_seq, alt_acceptor_seq =
                    reverse_complement(acceptor_seq),
                    reverse_complement(donor_seq),
                    reverse_complement(alt_acceptor_seq),
                    reverse_complement(alt_donor_seq)

                donor_cons_i, acceptor_cons_i, alt_donor_cons_i, alt_acceptor_cons_i =
                    reverse(acceptor_cons_i),
                    reverse(donor_cons_i),
                    reverse(alt_acceptor_cons_i),
                    reverse(alt_donor_cons_i)
            end

            donor_seqs[i]        = donor_seq
            acceptor_seqs[i]     = acceptor_seq
            alt_donor_seqs[i]    = alt_donor_seq
            alt_acceptor_seqs[i] = alt_acceptor_seq

            donor_cons[i,:] = donor_cons_i
            acceptor_cons[i,:] = acceptor_cons_i
            alt_donor_cons[i,:] = alt_donor_cons_i
            alt_acceptor_cons[i,:] = alt_acceptor_cons_i

            if i <= 3
                @show (i, intron.seqname)
                @show haskey(conservations, intron.seqname)
            end
        end
    end

    # 1-hot encode sequences
    donor_seq_arr        = one_hot_encode_seqs(donor_seqs)
    acceptor_seq_arr     = one_hot_encode_seqs(acceptor_seqs)
    alt_donor_seq_arr    = one_hot_encode_seqs(alt_donor_seqs)
    alt_acceptor_seq_arr = one_hot_encode_seqs(alt_acceptor_seqs)

    return donor_seq_arr, acceptor_seq_arr, alt_donor_seq_arr, alt_acceptor_seq_arr,
        donor_cons, acceptor_cons, alt_donor_cons, alt_acceptor_cons
end


function encode_kmer_usage!(usage_matrix, seq, cons, i, j)
    z = length(seq) - K + 1

    for (l, x) in each(DNAKmer{K}, seq)

        c = 0.0
        for i in l:l+K-1
            c += cons[i]
        end
        c /= K

        k = convert(UInt64, x)

        # usage_matrix[i, j + k] = 1
        # usage_matrix[i, j + k] += 1
        # usage_matrix[i, j + k] += c
        usage_matrix[i, j + k] = max(usage_matrix[i, j + k], c)

        # @show (l, c, x)
    end

    # exit()
end


function read_conservations(conservation_filename)
    conservation_input = h5open(conservation_filename)
    conservations = Dict{String, Vector{Float32}}()
    for dataset in get_datasets(conservation_input)
        chrname = replace(name(dataset), r"^/" => "")
        conservations[chrname] = read(dataset)

        # TODO: values tend to be too small. Maybe clamp the upper end?

        nancount = sum(map(isnan, conservations[chrname]))
        @show (chrname, nancount / length(conservations[chrname]))

        # but everything into a 0-1 scale. not sure this is a great idea
        replace!(x -> isnan(x) ? 0.0f0 : x, conservations[chrname])

        # clamp!(conservations[chrname], -1.0, 1.0)
        # # @show quantile(conservation[chrname], [0.0, 0.1, 0.5, 0.9, 1.0])
        # # map!(exp, conservation[chrname], conservation[chrname])
        # conservations[chrname] .-= minimum(conservations[chrname])
        # conservations[chrname] ./= maximum(conservations[chrname])
    end
    return conservations
end


function extract_sequence_kmer_features(
        genome_filename, conservation_filename, cassette_exons)

    conservations = read_conservation(conservation_filename)

    num_features = length(cassette_exons)

    num_segments = 7
    segment_size = 4^K
    kmer_usage_matrix = zeros(Float32, (num_features, num_segments*segment_size))

    # fill!(kmer_usage_matrix, 1)
    fill!(kmer_usage_matrix, 0)

    intron_segment_len = 50

    reader = open(FASTA.Reader, genome_filename)
    entry = eltype(reader)()
    while Polee.tryread!(reader, entry)
        seqname = FASTA.identifier(entry)
        entryseq = FASTA.sequence(entry)

        if !haskey(conservations, seqname)
            conservation = zeros(Float32, length(entryseq))
        else
            conservation = conservations[seqname]
        end

        for (i, (intron, flanks)) in enumerate(cassette_exons)
            if intron.seqname != seqname
                continue
            end

            exon_first = flanks.metadata[1]
            exon_last  = flanks.metadata[2]

            w = intron.first
            x = intron.first + intron_segment_len - 1
            y = exon_first - intron_segment_len
            z = exon_first - 1
            intron_5p_A = extract_sequence(entryseq, w, x)
            intron_5p_A_cons = conservation[w:x]

            intron_5p_B = extract_sequence(entryseq, x+1, y-1)
            intron_5p_B_cons = conservation[x+1:y-1]

            intron_5p_C = extract_sequence(entryseq, y, z)
            intron_5p_C_cons = conservation[y:z]

            alt_exon = extract_sequence(entryseq, exon_first, exon_last)
            alt_exon_cons = conservation[exon_first:exon_last]

            w = exon_last + 1
            x = exon_last + intron_segment_len
            y = intron.last - intron_segment_len + 1
            z = intron.last
            intron_3p_A = extract_sequence(entryseq, w, x)
            intron_3p_A_cons = conservation[w:x]

            intron_3p_B = extract_sequence(entryseq, x+1, y-1)
            intron_3p_B_cons = conservation[x+1:y-1]

            intron_3p_C = extract_sequence(entryseq, y, z)
            intron_3p_C_cons = conservation[y:z]

            if intron.strand == STRAND_NEG
                intron_5p_A_ = reverse_complement(intron_3p_C)
                intron_5p_B_ = reverse_complement(intron_3p_B)
                intron_5p_C_ = reverse_complement(intron_3p_A)

                intron_5p_A_cons_ = reverse(intron_3p_C_cons)
                intron_5p_B_cons_ = reverse(intron_3p_B_cons)
                intron_5p_C_cons_ = reverse(intron_3p_A_cons)

                intron_3p_A_ = reverse_complement(intron_5p_C)
                intron_3p_B_ = reverse_complement(intron_5p_B)
                intron_3p_C_ = reverse_complement(intron_5p_A)

                intron_3p_A_cons_ = reverse(intron_5p_C_cons)
                intron_3p_B_cons_ = reverse(intron_5p_B_cons)
                intron_3p_C_cons_ = reverse(intron_5p_A_cons)

                intron_5p_A = intron_5p_A_
                intron_5p_B = intron_5p_B_
                intron_5p_C = intron_5p_C_

                intron_5p_A_cons = intron_5p_A_cons_
                intron_5p_B_cons = intron_5p_B_cons_
                intron_5p_C_cons = intron_5p_C_cons_

                intron_3p_A = intron_3p_A_
                intron_3p_B = intron_3p_B_
                intron_3p_C = intron_3p_C_

                intron_3p_A_cons = intron_3p_A_cons_
                intron_3p_B_cons = intron_3p_B_cons_
                intron_3p_C_cons = intron_3p_C_cons_

                alt_exon = reverse_complement(alt_exon)
                reverse!(alt_exon_cons)
            end

            # TODO: encode (i.e. write)
            encode_kmer_usage!(kmer_usage_matrix, intron_5p_A, intron_5p_A_cons, i, 1 + 0*segment_size)
            encode_kmer_usage!(kmer_usage_matrix, intron_5p_B, intron_5p_B_cons, i, 1 + 1*segment_size)
            encode_kmer_usage!(kmer_usage_matrix, intron_5p_C, intron_5p_C_cons, i, 1 + 2*segment_size)
            encode_kmer_usage!(kmer_usage_matrix, alt_exon, alt_exon_cons, i, 1 + 3*segment_size)
            encode_kmer_usage!(kmer_usage_matrix, intron_3p_A, intron_3p_A_cons, i, 1 + 4*segment_size)
            encode_kmer_usage!(kmer_usage_matrix, intron_3p_B, intron_3p_B_cons, i, 1 + 5*segment_size)
            encode_kmer_usage!(kmer_usage_matrix, intron_3p_C, intron_3p_C_cons, i, 1 + 6*segment_size)
        end
    end

    # for idx in eachindex(kmer_usage_matrix)
    #     kmer_usage_matrix[idx] = log(kmer_usage_matrix[idx])
    # end

    return kmer_usage_matrix
end


function get_tissues_from_spec(spec)
    tissue_idx = Dict{String, Int}()
    tissues = Int[]
    for sample in spec["samples"]
        @assert length(sample["factors"]) == 1
        tissue = sample["factors"][1]

        k = get!(tissue_idx, tissue, 1 + length(tissue_idx))
        push!(tissues, k)
    end

    return tissues, tissue_idx
end


function main()
    # read specification
    spec = YAML.load_file(ARGS[1])
    conservation_filename = ARGS[2]
    if isempty(spec)
        error("Experiment specification is empty.")
    end

    if !haskey(spec, "samples")
        error("Experiment specification has no samples.")
    end

    if !haskey(spec, "genome")
        error("Experiment specification must have genome file specified.")
    end
    genome_filename = spec["genome"]

    excluded_transcripts = Set{String}()
    if length(ARGS) >= 3
        open(ARGS[3]) do input
            for line in eachline(input)
                push!(excluded_transcripts, chomp(line))
            end
        end
    end

    if haskey(spec, "annotations")
        transcripts_filename = spec["annotations"]
    else
        first_sample = first(spec["samples"])
        if haskey(first_sample, "file")
            first_sample_file = first_sample["file"]
        else
            if !haskey(first_sample, "name")
                error("Sample in experiment specification is missing a 'name' field.")
            end
            first_sample_file = string(first_sample["name"], prep_file_suffix)
        end

        transcripts_filename =
            Polee.read_transcripts_filename_from_prepared(first_sample_file)
        println("Using transcripts file: ", transcripts_filename)
    end

    tissues, tissue_idx = get_tissues_from_spec(spec)

    conservations = read_conservations(conservation_filename)

    Polee.init_python_modules()

    ts, ts_metadata = Polee.Transcripts(transcripts_filename, excluded_transcripts)
    # read_transcript_sequences!(ts, spec["genome"]) # don't need this

    max_num_samples = nothing
    batch_size = nothing

    loaded_samples = Polee.load_samples_from_specification(
        spec, ts, ts_metadata, max_num_samples, batch_size)

    # find splicing features
    cassette_exons, mutex_exons = Polee.get_cassette_exons(ts)
    # cassette_exons = filter(e -> e[1].strand == STRAND_POS, cassette_exons)

    @info string("Read ", length(cassette_exons), " cassette exons")

    qx_feature_loc, qx_feature_scale = approximate_splicing_likelihood(
        cassette_exons, loaded_samples)


    # qx_feature_scale_var =
    #     maximum(qx_feature_scale, dims=1)[1,:] .- minimum(qx_feature_scale, dims=1)[1,:]
    # p = sortperm(qx_feature_scale_var, rev=true)


    @show extrema(qx_feature_loc[:,1505])
    @show extrema(qx_feature_scale[:,1505])

    upperq = Float64[
        quantile(qx_feature_scale[:,i], 0.9) for i in 1:size(qx_feature_scale, 2)]
    lowerq = Float64[
        quantile(qx_feature_scale[:,i], 0.1) for i in 1:size(qx_feature_scale, 2)]
    p = sortperm(upperq .- lowerq, rev=true)


    # qx_feature_scale_var = var(qx_feature_scale, dims=1)[1,:]
    # p = sortperm(qx_feature_scale_var, rev=true)

    # qx_feature_scale_var = minimum(qx_feature_scale, dims=1)[1,:]
    # p = sortperm(qx_feature_scale_var)

    loc = qx_feature_loc[:,p]
    scale = qx_feature_scale[:,p]

    open("qx_feature_loc.csv", "w") do output
        for j in 1:size(loc, 2)
            print(output, loc[1, j])
            for i in 2:size(loc, 1)
                print(output, ", ", loc[i, j])
            end
            println(output)
        end
    end

    open("qx_feature_scale.csv", "w") do output
        for j in 1:size(scale, 2)
            print(output, scale[1, j])
            for i in 2:size(scale, 1)
                print(output, ", ", scale[i, j])
            end
            println(output)
        end
    end

    # for i in 1:40
    #     println((i, p[i]))
    # end

    # exit()


    # qx_feature_loc = qx_feature_loc[:,p][:,1:250]
    # qx_feature_scale = qx_feature_scale[:,p][:,1:250]
    # cassette_exons = cassette_exons[p][1:250]


    # CNN using small amount of sequence around splice junctions
    donor_seqs, acceptor_seqs, alt_donor_seqs, alt_acceptor_seqs,
    donor_cons, acceptor_cons, alt_donor_cons, alt_acceptor_cons =
        extract_sequence_features(genome_filename, conservations, cassette_exons)

    # simple regression using k-mer presence
    # kmer_usage_matrix = extract_sequence_kmer_features(
    #     genome_filename, conservation_filename, cassette_exons)


    # @show size(qx_feature_loc)
    # @show size(qx_feature_scale)

    # tmp_mu = qx_feature_loc[1:4,:]
    # tmp_sd = qx_feature_scale[1:4,:]

    # tmp_sd_var = var(tmp_sd, dims=1)[1,:]

    # p = sortperm(tmp_sd_var, rev=true)
    # tmp_mu = tmp_mu[:,p]
    # tmp_sd = tmp_sd[:,p]

    # @show tmp_sd_var[p][1:10]
    # for i in 1:10
    #     println(i)
    #     println(tmp_mu[:,i])
    #     println(tmp_sd[:,i])
    # end

    # @show tmp_sd_var[p][100:110]
    # for i in 100:110
    #     println(i)
    #     println(tmp_mu[:,i])
    #     println(tmp_sd[:,i])
    # end

    # exit()

    # println("---------------")

    # k = div(size(tmp_mu, 2), 2)
    # for i in 1:10
    #     println(i)
    #     println(tmp_mu[:,k+i])
    #     println(tmp_sd[:,k+i])
    # end

    # println("---------------")


    # for i in 1:10
    #     println(i)
    #     println(tmp_mu[:,end-i+1])
    #     println(tmp_sd[:,end-i+1])
    # end

    # println("#####################")

    # p = sortperm(tmp_sd[1,:])
    # tmp_mu = tmp_mu[:,p]
    # tmp_sd = tmp_sd[:,p]

    # for i in 1:10
    #     println(i)
    #     println(tmp_mu[:,i])
    #     println(tmp_sd[:,i])
    # end

    # @show quantile(tmp_sd[1,:], [0.0, 0.1, 0.5, 0.9, 1.0])

    # exit()

    splice_code_py.estimate_splicing_code(
        qx_feature_loc, qx_feature_scale,
        donor_seqs, acceptor_seqs, alt_donor_seqs, alt_acceptor_seqs,
        donor_cons, acceptor_cons, alt_donor_cons, alt_acceptor_cons,
        tissues)

    # W0, W = splice_code_py.estimate_splicing_code_from_kmers(
    #     qx_feature_loc, qx_feature_scale,
    #     kmer_usage_matrix, tissues)

    # print_top_k_ws(W0, 10)

    # println("------------")

    # p = sortperm(var(W, dims=2)[:,1], rev=true)

    # for i in 1:5
    #     k = p[i]-1
    #     while k >= 4^K
    #         k -= 4^K
    #     end
    #     @show DNAKmer{K}(UInt64(k))
    #     @show W[p[i],:]
    # end

    # @show tissue_idx
end


function print_top_k_ws(W, k)
    segment_size = 4^K
    segments = [
        "intron_5p_A",
        "intron_5p_B",
        "intron_5p_C",
        "exon",
        "intron_3p_A",
        "intron_3p_B",
        "intron_3p_C" ]

    for j in 1:size(W, 2)
        println("tissue: ", j)
        for (i, part) in enumerate(segments)
            println(part)

            from = 1 + (i-1)*segment_size
            to = (i)*segment_size

            W_part = W[from:to,j]
            p = sortperm(abs.(W_part), rev=true)

            for i in 1:k
                println((DNAKmer{K}(UInt64(p[i] - 1)), W_part[p[i]]))
            end
        end

        # for i in 1:k
        #     println((DNAKmer{6}(UInt64(p[i] - 1)), W[p[i], j]))
        # end
    end
end


main()

