
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


function extract_sequence_features(genome_filename, cassette_exons)

    num_features = length(cassette_exons)

    donor_seqs        = Array{DNASequence}(undef, num_features)
    acceptor_seqs     = Array{DNASequence}(undef, num_features)
    alt_donor_seqs    = Array{DNASequence}(undef, num_features)
    alt_acceptor_seqs = Array{DNASequence}(undef, num_features)

    exonic_len = 15
    intronic_len = 15

    reader = open(FASTA.Reader, genome_filename)
    entry = eltype(reader)()
    while Polee.tryread!(reader, entry)
        seqname = FASTA.identifier(entry)
        entryseq = FASTA.sequence(entry)
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

            if intron.strand == STRAND_NEG
                donor_seq, acceptor_seq, alt_donor_seq, alt_acceptor_seq =
                    reverse_complement(acceptor_seq),
                    reverse_complement(donor_seq),
                    reverse_complement(alt_acceptor_seq),
                    reverse_complement(alt_donor_seq)
            end

            donor_seqs[i]        = donor_seq
            acceptor_seqs[i]     = acceptor_seq
            alt_donor_seqs[i]    = alt_donor_seq
            alt_acceptor_seqs[i] = alt_acceptor_seq
        end
    end

    # 1-hot encode sequences
    donor_seq_arr        = one_hot_encode_seqs(donor_seqs)
    acceptor_seq_arr     = one_hot_encode_seqs(acceptor_seqs)
    alt_donor_seq_arr    = one_hot_encode_seqs(alt_donor_seqs)
    alt_acceptor_seq_arr = one_hot_encode_seqs(alt_acceptor_seqs)

    return donor_seq_arr, acceptor_seq_arr, alt_donor_seq_arr, alt_acceptor_seq_arr
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

    Polee.init_python_modules()

    excluded_transcripts = Set{String}()
    ts, ts_metadata = Polee.Transcripts(transcripts_filename, excluded_transcripts)
    # read_transcript_sequences!(ts, spec["genome"]) # don't need this

    max_num_samples = nothing
    batch_size = nothing

    loaded_samples = Polee.load_samples_from_specification(
        spec, ts, ts_metadata, max_num_samples, batch_size)

    # find splicing features
    cassette_exons, mutex_exons = Polee.get_cassette_exons(ts)
    @info string("Read ", length(cassette_exons), " cassette exons")

    donor_seqs, acceptor_seqs, alt_donor_seqs, alt_acceptor_seqs =
        extract_sequence_features(genome_filename, cassette_exons)

    qx_feature_loc, qx_feature_scale = approximate_splicing_likelihood(
        cassette_exons, loaded_samples)

    splice_code_py.estimate_splicing_code(
        qx_feature_loc, qx_feature_scale,
        donor_seqs, acceptor_seqs, alt_donor_seqs, alt_acceptor_seqs,
        tissues)
end


main()

