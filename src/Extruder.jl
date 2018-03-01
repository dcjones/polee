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
include("sinh_arcsinh.jl")
include("likelihood-approximation.jl")
include("estimate.jl")
include("gibbs.jl")
include("stick_breaking.jl")
include("isometric_log_ratios.jl")
include("additive_log_ratios.jl")
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


function select_approx_method(method_name::String, tree_method::Symbol)
    if method_name == "optimize"
        return OptimizeHSBApprox()
    elseif method_name == "logistic_normal"
        return LogisticNormalApprox()
    elseif method_name == "kumaraswamy_hsb"
        return KumaraswamyHSBApprox(tree_method)
    elseif method_name == "logit_skew_normal_hsb"
        return LogitSkewNormalHSBApprox(tree_method)
    elseif method_name == "logit_normal_hsb"
        return LogitNormalHSBApprox(tree_method)
    elseif method_name == "normal_ilr"
        return NormalILRApprox(tree_method)
    elseif method_name == "normal_alr"
        return NormalALRApprox()
    else
        error("$(method_name) is not a know approximation method.")
    end
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
            "--exclude-transcripts"
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

        excluded_transcripts = Set{String}()
        if parsed_args["exclude-transcripts"] != nothing
            open(parsed_args["exclude-transcripts"]) do input
                for line in eachline(input)
                    push!(excluded_transcripts, chomp(line))
                end
            end
        end

        sample = RNASeqSample(parsed_args["transcripts_filename"],
                              parsed_args["genome_filename"],
                              parsed_args["reads_filename"],
                              excluded_seqs,
                              excluded_transcripts,
                              Nullable{String}(parsed_args["output"]))
        return

    elseif subcmd == "likelihood-approx"
        @add_arg_table arg_settings begin
            "--output", "-o"
                default = "sample-data.h5"
            "--approx-method"
                default = "logit_skew_normal_hsb"
            "--tree-method"
                default = "cluster"
            "likelihood_matrix_filename"
                required = true
        end
        parsed_args = parse_args(subcmd_args, arg_settings)

        tree_method = Symbol(parsed_args["tree-method"])
        approx = select_approx_method(parsed_args["approx-method"], tree_method)
        approximate_likelihood(approx,
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
            "--exclude-seqs"
                required = false
            "--exclude-transcripts"
                required = false
            "--likelihood-matrix"
                required = false
            "--approx-method"
                default = "logit_skew_normal_hsb"
            "--tree-method"
                default = "cluster"
        end
        parsed_args = parse_args(subcmd_args, arg_settings)

        tree_method = Symbol(parsed_args["tree-method"])
        approx = select_approx_method(parsed_args["approx-method"], tree_method)

        excluded_seqs = Set{String}()
        if parsed_args["exclude-seqs"] != nothing
            open(parsed_args["exclude-seqs"]) do input
                for line in eachline(input)
                    push!(excluded_seqs, chomp(line))
                end
            end
        end

        excluded_transcripts = Set{String}()
        if parsed_args["exclude-transcripts"] != nothing
            open(parsed_args["exclude-transcripts"]) do input
                for line in eachline(input)
                    push!(excluded_transcripts, chomp(line))
                end
            end
        end

        sample = RNASeqSample(parsed_args["transcripts_filename"],
                              parsed_args["genome_filename"],
                              parsed_args["reads_filename"],
                              excluded_seqs,
                              excluded_transcripts,
                              parsed_args["likelihood-matrix"] == nothing ?
                                Nullable{String}() :
                                Nullable(parsed_args["likelihood-matrix"]))
        approximate_likelihood(approx, sample, parsed_args["output"])
        return

    elseif subcmd == "estimate" || subcmd == "est"
        @add_arg_table arg_settings begin
            "--output", "-o"
                default = Nullable{String}()
            "--output-format", "-F"
                default = "csv"
            "--exclude-transcripts"
                required = false
            "--credible-lower"
                default = 0.025
                arg_type = Float64
            "--credible-upper"
                default = 0.975
                arg_type = Float64
            "--inference"
                default = "variational"
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

        excluded_transcripts = Set{String}()
        if parsed_args["exclude-transcripts"] != nothing
            open(parsed_args["exclude-transcripts"]) do input
                for line in eachline(input)
                    push!(excluded_transcripts, chomp(line))
                end
            end
        end

        ts, ts_metadata = Transcripts(parsed_args["transcripts"], excluded_transcripts)
        gene_db = write_transcripts("genes.db", ts, ts_metadata)

        loaded_samples =
            load_samples_from_specification(parsed_args["experiment"], ts, ts_metadata)

        inference = Symbol(parsed_args["inference"])
        feature = Symbol(parsed_args["feature"])

        output_format = Symbol(parsed_args["output-format"])
        if output_format != :csv && output_format != :sqlite3
            error("Output format must be either \"csv\" or \"sqlite3\".")
        end

        credible_interval =
            (Float64(parsed_args["credible-lower"]),
             Float64(parsed_args["credible-upper"]))

        input = ModelInput(
            loaded_samples, inference, feature, ts, ts_metadata,
            parsed_args["output"], output_format, gene_db,
            credible_interval)

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
    elseif subcmd == "approx-sample"
        @add_arg_table arg_settings begin
            "--output", "-o"
                default = "post_mean.csv"
            "--exclude-transcripts"
                required = false
            "--num-samples"
                default = 1000
                arg_type = Int
            "transcripts"
                required = true
            "prepared_sample"
                required = true
        end
        parsed_args = parse_args(subcmd_args, arg_settings)

        excluded_transcripts = Set{String}()
        if parsed_args["exclude-transcripts"] != nothing
            open(parsed_args["exclude-transcripts"]) do input
                for line in eachline(input)
                    push!(excluded_transcripts, chomp(line))
                end
            end
        end

        ts, ts_metadata = Transcripts(parsed_args["transcripts"], excluded_transcripts)
        n = length(ts)

        input = h5open(parsed_args["prepared_sample"])
        node_parent_idxs = read(input["node_parent_idxs"])
        node_js          = read(input["node_js"])
        efflens          = read(input["effective_lengths"])

        mu    = read(input["mu"])
        sigma = exp.(read(input["omega"]))
        alpha = read(input["alpha"])

        t = HSBTransform(node_parent_idxs, node_js)

        num_samples = parsed_args["num-samples"]
        samples = Array{Float32}((num_samples, n))

        zs0 = Array{Float32}(n-1)
        zs = Array{Float32}(n-1)
        ys = Array{Float64}(n-1)
        xs = Array{Float32}(n)

        prog = Progress(num_samples, 0.25, "Sampling from approx. likelihood ", 60)
        for i in 1:num_samples
            next!(prog)
            for j in 1:n-1
                zs0[j] = randn(Float32)
            end
            sinh_asinh_transform!(alpha, zs0, zs, Val{true})
            logit_normal_transform!(mu, sigma, zs, ys, Val{true})
            hsb_transform!(t, ys, xs, Val{true})

            # effective length transform
            xs ./= efflens
            xs ./= sum(xs)
            samples[i, :] = xs
        end
        finish!(prog)

        # TODO: only interested in the mean for now, but we may want to dump
        # all the samples some time.
        post_mean = mean(samples, 1)
        @show size(post_mean)
        open(parsed_args["output"], "w") do output
            for j in 1:length(post_mean)
                println(output, post_mean[j])
            end
        end

        return
    else
        println("Unknown command: ", subcmd, "\n")
        print_usage()
        exit(1)
    end
end

end # module Extruder
