#!/usr/bin/env julia

module Extruder

using ArgParse
using GenomicFeatures
using BioAlignments
using BioSequences
using DataStructures
using Distributions
using HATTries
using HDF5
using ProgressMeter
using PyCall
using SQLite
using StatsBase
import IntervalTrees
import SHA
import YAML

# import TensorFlow

include("constants.jl")
include("sparse.jl")
include("transcripts.jl")
include("reads.jl")
include("bias.jl")
include("fragmodel.jl")
include("model.jl")
include("sample.jl")
include("kumaraswamy.jl")
include("logitnormal.jl")
include("likelihood-approximation.jl")
include("estimate.jl")
include("gibbs.jl")
include("stick_breaking.jl")
include("isometric_log_ratios.jl")
include("sequences.jl")
include("evaluate.jl")

# TODO: automate including everything under models
EXTRUDER_MODELS = Dict{String, Function}()
include("models/linear-regression.jl")
include("models/simple-linear-regression.jl")
include("models/simple-mode.jl")
include("models/logistic-regression.jl")
include("models/simple-logistic-regression.jl")
include("models/quantification.jl")
include("models/pca.jl")


function print_usage()
    println("Usage: extruder <command>\n")
    println("where command is one of:")
    println("  likelihood-matrix")
    println("  likelihood-approx")
    println("  prepare-sample")
    println("  estimate")
end


function main()
    srand(12345678)

    if isempty(ARGS)
        print_usage()
        exit(1)
    end

    subcmd = ARGS[1]
    subcmd_args = ARGS[2:end]
    arg_settings = ArgParseSettings()

    if subcmd == "likelihood-matrix"
        @add_arg_table arg_settings begin
            "--output", "-o"
                default = "likelihood-matrix.h5"
            "transcripts_filename"
                required = true
            "genome_filename"
                required = true
            "reads_filename"
                required = true
            "--excluded-seqs"
                required = false
        end
        parsed_args = parse_args(subcmd_args, arg_settings)

        excluded_seqs = Set{String}()
        if parsed_args["excluded-seqs"] != nothing
            open(parsed_args["excluded-seqs"]) do input
                for line in eachline(input)
                    push!(excluded_seqs, chop(line))
                end
            end
        end

        sample = RNASeqSample(parsed_args["transcripts_filename"],
                              parsed_args["genome_filename"],
                              parsed_args["reads_filename"],
                              excluded_seqs,
                              Nullable(parsed_args["output"]))
        return

    elseif subcmd == "likelihood-approx"
        @add_arg_table arg_settings begin
            "--output", "-o"
                default = "sample-data.h5"
            "likelihood_matrix_filename"
                required = true
        end
        parsed_args = parse_args(subcmd_args, arg_settings)
        # TODO: approx method from command line
        approximate_likelihood(LogitNormalHSBApprox(:sequential),
                               parsed_args["likelihood_matrix_filename"],
                               parsed_args["output"])
        return

    elseif subcmd == "prepare-sample" || subcmd == "prep"
        @add_arg_table arg_settings begin
            "--output", "-o"
                default = "sample-data.h5"
            "transcripts_filename"
                required = true
            "genome_filename"
                required = true
            "reads_filename"
                required = true
            "--excluded-seqs"
                required = false
            "--likelihood-matrix"
                required = false
        end
        parsed_args = parse_args(subcmd_args, arg_settings)

        excluded_seqs = Set{String}()
        if parsed_args["excluded-seqs"] != nothing
            open(parsed_args["excluded-seqs"]) do input
                for line in eachline(input)
                    push!(excluded_seqs, chop(line))
                end
            end
        end

        sample = RNASeqSample(parsed_args["transcripts_filename"],
                              parsed_args["genome_filename"],
                              parsed_args["reads_filename"],
                              excluded_seqs,
                              parsed_args["likelihood-matrix"] == nothing ?
                                Nullable{String}() :
                                Nullable(parsed_args["likelihood-matrix"]))
        approximate_likelihood(sample, parsed_args["output"])
        return

    elseif subcmd == "estimate" || subcmd == "est"
        @add_arg_table arg_settings begin
            "--output", "-o"
                default = Nullable{String}()
            "feature"
                required = true
            "model"
                required = true
            "transcripts"
                required = true
            "experiment"
                required = true
        end

        parsed_args = parse_args(subcmd_args, arg_settings)

        ts, ts_metadata = Transcripts(parsed_args["transcripts"])
        gene_db = write_transcripts("genes.db", ts, ts_metadata)

        (likapprox_musigma, likapprox_efflen, likapprox_As,
         likapprox_parent_idxs, likapprox_js, x0, sample_factors, sample_names) =
            load_samples_from_specification(parsed_args["experiment"], ts_metadata)

        feature = Symbol(parsed_args["feature"])

        input = ModelInput(
            likapprox_musigma, likapprox_efflen, likapprox_As,
            likapprox_parent_idxs, likapprox_js, x0, sample_factors,
            sample_names, feature, ts, ts_metadata,
            parsed_args["output"], gene_db)

        EXTRUDER_MODELS[parsed_args["model"]](input)

        # TODO: figure out what to do with `output`

        return
    elseif subcmd == "sample"
        @add_arg_table arg_settings begin
            "--output", "-o"
                default = "samples.csv"
            "likelihood_matrix"
                required = true
        end
        parsed_args = parse_args(subcmd_args, arg_settings)
        gibbs_sampler(parsed_args["likelihood_matrix"],
                      parsed_args["output"])
        return
    else
        println("Unknown command: ", subcmd, "\n")
        print_usage()
        exit(1)
    end
end


main()

end
