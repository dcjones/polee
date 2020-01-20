#!/usr/bin/env julia

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
import Polee

include(joinpath(dirname(pathof(Polee)), "PoleeModel.jl"))
using .PoleeModel


using ArgParse
using CSV
using YAML
using Statistics
using DataFrames
using Distributions
using StatsFuns
using Random
using DecisionTree
using DelimitedFiles
using SparseArrays
using LinearAlgebra: I
using HDF5


const arg_settings = ArgParseSettings()
arg_settings.prog = "evaluate-regression-results.jl"
@add_arg_table arg_settings begin
    "--output"
        metavar = "filename"
        default = "regression-evaluation.csv"
    "--num-samples"
        metavar = "N"
        help = "Number of samples to draw when evaluating entropy."
        default = 50
        arg_type = Int
    "--effect-size"
        metavar = "S"
        help = "Output the posterior probability of abs log2 fold-change greater than S"
        default = 1.5
        arg_type = Float64
    "--credible-interval"
        metavar = "C"
        help = """Size of the 0-centered credible interval to use when estimating
            minimum effect size."""
        default = 0.1
        arg_type = Float64
    "factor"
        help = "Factors to regress on."
        arg_type = String
    "regression-results"
        metavar = "regression-results.csv"
        help = "Results table from regression model"
    "experiment"
        metavar = "experiment.yml"
        help = "Experiment specification for testing data."
end


function find_minimum_effect_size(μ, σ, target_coverage)
    dist = Normal(μ, σ)

    δ_min = 0.0
    δ_max = 20.0
    coverage = 1.0
    while abs(coverage - target_coverage) > 0.001
        δ = (δ_max + δ_min) / 2
        coverage = cdf(dist, δ) - cdf(dist, -δ)

        if coverage > target_coverage
            δ_max = δ
        else
            δ_min = δ
        end
    end

    δ = (δ_max + δ_min) / 2
    return δ
end


function main()
    parsed_args = parse_args(arg_settings)

    reg = CSV.read(parsed_args["regression-results"])

    ts, ts_metadata = load_transcripts_from_args(
        parsed_args, experiment_arg="experiment")

    testing_spec = YAML.load_file(parsed_args["experiment"])

    _, testing_sample_names, testing_sample_factors =
        PoleeModel.read_specification(testing_spec)

    x_samplers_testing, testing_efflens, testing_sample_names, testing_sample_factors =
            load_samplers_from_specification(testing_spec, ts, ts_metadata)

    num_test_samples = length(x_samplers_testing)

    tnames = String[t.metadata.name for t in ts]
    x_perm = sortperm(tnames)
    tnames = tnames[x_perm]

    qx_bias_loc_df = unstack(
        reg[:,[:transcript_id, :factor, :qx_bias_loc]],
        :transcript_id, :factor, :qx_bias_loc)
    sort!(qx_bias_loc_df, :transcript_id)
    qx_bias_loc = Matrix{Float64}(qx_bias_loc_df[:,2:end])
    qx_bias_loc = qx_bias_loc[:,1]

    qx_scale_df = unstack(
        reg[:,[:transcript_id, :factor, :qx_scale]],
        :transcript_id, :factor, :qx_scale)
    sort!(qx_scale_df, :transcript_id)
    qx_scale = Matrix{Float64}(qx_scale_df[:,2:end])
    qx_scale = qx_scale[:,1]

    qw_loc_df = unstack(
        reg[:,[:transcript_id, :factor, :qw_loc]],
        :transcript_id, :factor, :qw_loc)
    sort!(qw_loc_df, :transcript_id)
    qw_loc = Matrix{Float64}(qw_loc_df[:,2:end])

    qw_scale_df = unstack(
        reg[:,[:transcript_id, :factor, :qw_scale]],
        :transcript_id, :factor, :qw_scale)
    sort!(qw_scale_df, :transcript_id)
    qw_scale = Matrix{Float64}(qw_scale_df[:,2:end])

    @assert size(qw_loc) == size(qw_scale)

    n = length(qx_bias_loc)

    factor_idx = Dict{String, Int}()
    for (i, name) in enumerate(names(qx_bias_loc_df)[2:end])
        factor_idx[string(name)] = i
    end
    factor_names = Vector{String}(map(string, names(qx_bias_loc_df)[2:end]))
    num_classes = length(factor_idx)

    sample_classes = [string(parsed_args["factor"], ":", s[parsed_args["factor"]])
        for s in testing_sample_factors]

    sample_class_idxs = Int[]
    for c in sample_classes
        push!(sample_class_idxs, factor_idx[c])
    end

    xs = zeros(Float64, (num_test_samples, n))
    scales = ones(Float64, num_test_samples)

    effect_size = log(abs(parsed_args["effect-size"]))
    class_probs = zeros(Float64, num_classes)
    true_label_probs = zeros(Float64, (num_classes, n))

    dist = TDist(20.0)
    for t in 1:parsed_args["num-samples"]
        println(t, "/", parsed_args["num-samples"])
        draw_samples!(x_samplers_testing, testing_efflens, xs, x_perm)

        high_expr_idx = qx_bias_loc .> quantile(qx_bias_loc, 0.9)
        sample_scales = median(
            (xs .- reshape(qx_bias_loc, (1, n)))[:,high_expr_idx],
            dims=2)[:,1]

        xs .-= sample_scales

        for i in 1:num_test_samples, j in 1:n
            for k in 1:num_classes
                # using TDist to avoid zero probabilities
                class_probs[k] = pdf(
                    dist,
                    (xs[i, j] - (qx_bias_loc[j] + qw_loc[j, k])) / qx_scale[j])
            end

            class_probs_sum = sum(class_probs)

            # true label probability
            true_label_probs[sample_class_idxs[i], j] +=
                class_probs_sum == 0.0 ? 1/num_classes :
                class_probs[sample_class_idxs[i]] / class_probs_sum
        end
    end

    # average across sampler draws
    true_label_probs ./= parsed_args["num-samples"]

    # average across number of samples in each class
    class_counts = zeros(Int, num_classes)
    for idx in sample_class_idxs
        class_counts[idx] += 1
    end
    true_label_probs ./= class_counts

    credibility = parsed_args["credible-interval"]
    open(parsed_args["output"], "w") do output
        println(output, "transcript_id,factor,de_prob,min_effect_size,true_label_prob")
        for i in 1:num_classes, j in 1:n
            prob_down = cdf(dist, (-effect_size - qw_loc[j,i]) / qw_scale[j,i])
            prob_up = ccdf(dist, (effect_size - qw_loc[j,i]) / qw_scale[j,i])
            prob_de = prob_down + prob_up

            min_effect_size = find_minimum_effect_size(
                qw_loc[j,i], qw_scale[j,i], credibility)

            # change to log2 scale
            min_effect_size / log(2.0)

            println(
                output,
                tnames[j], ",",
                factor_names[i], ",",
                prob_de, ",",
                min_effect_size, ",",
                true_label_probs[i, j])
        end
    end
end


"""
Draw sample from approximated likelihood, adjust for effective length and log
transform.
"""
function draw_samples!(samplers, efflens, xs, x_perm)
    num_samples = length(samplers)
    Threads.@threads for i in 1:size(xs, 1)
        xs_row = @view xs[i,:]
        rand!(samplers[i], xs_row)
        xs_row ./= @view efflens[i,:]
        xs_row ./= sum(xs_row)
        permute!(xs_row, x_perm)
        map!(log, xs_row, xs_row)
    end
end

main()

