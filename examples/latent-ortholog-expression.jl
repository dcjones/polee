#!/usr/bin/env julia

# This program written to analyze a specific experiment involving comparisons
# across species. Is to place samples in the same latent space representing
# expression of orthologous groups of transcripts. Orthologs were identified
# using OrthoMCL.

import Polee
using PyCall
using YAML

@pyimport tensorflow as tf
@pyimport edward as ed
@pyimport edward.models as edmodels

function main()
    ortholog_groups_filename = ARGS[1]

    species_loaded_samples       = Polee.LoadedSamples[]
    species_transcripts          = Polee.Transcripts[]
    species_transcripts_metadatas = Polee.TranscriptsMetadata[]

    # Load samples and transcripts for every species
    num_samples = 0
    i = 2
    while i <= length(ARGS)
        if i + 1 > length(ARGS)
            error("Last gff filename needs to be matches with an experiment")
        end
        genes_filename = ARGS[i]
        experiment_filename = ARGS[i+1]

        ts, ts_metadata = Polee.Transcripts(genes_filename)

        spec = YAML.load_file(experiment_filename)
        loaded_samples = Polee.load_samples_from_specification(
            spec, ts, ts_metadata)

        push!(species_loaded_samples, loaded_samples)
        push!(species_transcripts, ts)
        push!(species_transcripts_metadatas, ts_metadata)

        num_samples += length(loaded_samples.sample_names)

        i += 2
    end
    num_taxons = length(species_loaded_samples)

    # Load orthoMCL ortholog groups
    ortholog_groups = Set{String}[]
    ortholog_group_by_id = Dict{String, Int}()
    taxons = Set{String}()
    for line in eachline(ortholog_groups_filename)
        group_name, transcripts_str = split(line, '\t')
        transcripts = split(transcripts_str, ',')

        # remove taxon indicators from names, exclude groups that don't
        # have transcripts from all species.
        empty!(taxons)
        for i in 1:length(transcripts)
            mat = match(r"^([^\|]*)\|(.*)$", transcripts[i])
            taxon = mat.captures[1]
            id = mat.captures[2]
            transcripts[i] = id
            push!(taxons, taxon)
        end

        if length(taxons) != num_taxons
            continue
        end

        group_idx = 1 + length(ortholog_groups)
        push!(ortholog_groups, Set(transcripts))
        for transcript_ in transcripts
            transcript = String(transcript_)
            @assert !haskey(ortholog_group_by_id, transcript)
            ortholog_group_by_id[transcript] = group_idx
        end
    end
    num_orth_groups = length(ortholog_groups)
    println("Read ", num_orth_groups, " ortholog groups")

    # for (k, (ts, ts_metadata)) in enumerate(zip(species_transcripts, species_transcripts_metadatas))
        # species_ortholog_matrix(
        #     num_orth_groups, ortholog_group_by_id, ts, ts_metadata)
    # end

    # Let' just start with point estimates for now.
    x_og_mu0 = log(1/num_orth_groups)
    x_og_sigma0 = 5.0
    x_og = edmodels.Normal(
        loc=tf.fill([num_samples, num_orth_groups], x_og_mu0),
        scale=x_og_sigma0)

    # slice x_og up by species
    idxoff = 0
    x_ogs = Any[]
    for k in 1:num_taxons
        ts = species_transcripts[k]
        ts_metadata = species_transcripts_metadatas[k]

        species_num_samples = length(species_loaded_samples[k].sample_names)
        x_og_k = tf.slice(x_og, [idxoff, 0], [species_num_samples, num_orth_groups])
        idxoff += species_num_samples

        orthgroup_to_grouped_transcript, grouped_transcript_to_transcript =
            species_ortholog_matrix(
                num_orth_groups, ortholog_group_by_id, ts, ts_metadata)

        x_gt = tf.transpose(tf.sparse_tensor_dense_matmul(orthgroup_to_grouped_transcript, tf.transpose(x_og_k)))
        @show x_gt

        # TODO: intra-group transcript mixtures

        # TODO: non-grouped expression

    end



    # TODO: set up intra-feature mixture variables.

    # TODO: feed that shit into approx likelihoods (again, seperate for each
    # species)
end


function species_ortholog_matrix(num_orth_groups, ortholog_group_by_id, ts, ts_metadata)
    n = length(ts)

    I = Int[]
    J = Int[]
    tid_to_grptid = Dict{Int, Int}()
    grptid_to_tid = Dict{Int, Int}()
    for t in ts
        if haskey(ortholog_group_by_id, t.metadata.name)
            push!(J, ortholog_group_by_id[t.metadata.name])
            @assert !haskey(tid_to_grptid, t.metadata.id)
            tid_to_grptid[t.metadata.id] = length(tid_to_grptid) + 1
            grptid_to_tid[tid_to_grptid[t.metadata.id]] = t.metadata.id
            push!(I, tid_to_grptid[t.metadata.id])
        end
    end

    # sparse matrix mapping ortholog group index to grouped transcript index
    orthgroup_to_grouped_transcript_indices =
        Array{Int32}((length(I), 2))
    for (k, (i, j)) in enumerate(zip(I, J))
        orthgroup_to_grouped_transcript_indices[k, 1] = i - 1
        orthgroup_to_grouped_transcript_indices[k, 2] = j - 1
    end

    orthgroup_to_grouped_transcript = tf.SparseTensor(
        indices=orthgroup_to_grouped_transcript_indices,
        values=tf.ones(length(I)),
        dense_shape=[length(I), num_orth_groups])

    # sparse matrix mapping grouped transcript index to transcript index
    grouped_transcript_to_transcript_indices =
        Array{Int32}((length(I), 2))
    for (k, (i, j)) in enumerate(zip(I, J))
        grouped_transcript_to_transcript_indices[k, 1] = i - 1
        grouped_transcript_to_transcript_indices[k, 2] = grptid_to_tid[i] - 1
    end

    grouped_transcript_to_transcript = tf.SparseTensor(
        indices=grouped_transcript_to_transcript_indices,
        values=tf.ones(length(I)),
        dense_shape=[n, length(I)])

    return orthgroup_to_grouped_transcript, grouped_transcript_to_transcript
end

main()
