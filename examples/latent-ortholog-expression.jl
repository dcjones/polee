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
    print("Read ", num_orth_groups, " ortholog groups")

    # TODO: construct a sparse feature matrix for each species, similar to the
    # one used for genes. Make sure to add aux groups for each ungrouped transcript

    # TODO: set up intra-feature mixture variables.

    # TODO: feed that shit into approx likelihoods (again, seperate for each
    # species)
end

main()
