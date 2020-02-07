#!/usr/bin/env julia

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
import Polee

include(joinpath(dirname(pathof(Polee)), "PoleeModel.jl"))
using .PoleeModel


using ArgParse
using CSV
using YAML
using Statistics
using StatsBase
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
    "--pseudocount"
        metavar = "C"
        default = 0.0
        help = "If specified with --point-estimates, add C tpm to each value."
        arg_type = Float64
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
    "--point-estimates"
        help = """
            Use point estimates read from a file specified in the experiment
            instead of approximated likelihood."""
        default = nothing
        arg_type = String
    "--kallisto-bootstrap"
        help = """
            Use kallisto bootstrap samples. The sample specifications should
            have a `kallisto` key pointing to the h5 file.
            """
        action = :store_true
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


"""
Find a value δ, where `target_coverage` of the probability mass is in (-δ, δ).
"""
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


function estimate_sample_scales(xs, qx_bias_loc, upper_quantile=0.9)
    high_expr_idx = qx_bias_loc .> quantile(qx_bias_loc, upper_quantile)
    n = length(qx_bias_loc)
    sample_scales = median(
        (xs .- reshape(qx_bias_loc, (1, n)))[:,high_expr_idx],
        dims=2)[:,1]
    return sample_scales
end


"""
Evaluate classifier accuracy using samples drawn from approximated likelihood.
"""
function evaluate_regression_with_likelihood_samples(
        testing_spec, ts, ts_metadata, x_perm, num_eval_samples, sample_class_idxs,
        qx_bias_loc, qw_loc, qx_scale)

    x_samplers_testing, testing_efflens, testing_sample_names, testing_sample_factors =
            load_samplers_from_specification(testing_spec, ts, ts_metadata)

    n = length(ts)
    num_classes = size(qw_loc, 2)
    num_test_samples = length(x_samplers_testing)
    xs = zeros(Float64, (num_test_samples, n))
    class_probs = zeros(Float64, num_classes)
    true_label_probs = zeros(Float64, (num_classes, n))

    # using TDist to avoid zero probabilities
    dist = TDist(20.0)
    for t in 1:num_eval_samples
        println(t, "/", num_eval_samples)
        draw_samples!(x_samplers_testing, testing_efflens, x_perm, xs)

        sample_scales = estimate_sample_scales(xs, qx_bias_loc)
        xs .-= sample_scales

        for i in 1:num_test_samples, j in 1:n
            for k in 1:num_classes
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
    true_label_probs ./= num_eval_samples

    # average across number of samples in each class
    class_counts = zeros(Int, num_classes)
    for idx in sample_class_idxs
        class_counts[idx] += 1
    end
    true_label_probs ./= class_counts

    return true_label_probs
end


"""
Evaluate classifier accuracy using point estimates.
"""
function evaluate_regression_with_point_estimates(
        testing_spec, ts, ts_metadata, x_perm, point_estimates_key, pseudocount,
        sample_class_idxs, qx_bias_loc, qw_loc, qx_scale)

    loaded_samples = load_point_estimates_from_specification(
        testing_spec, ts, ts_metadata, point_estimates_key)

    xs = log.(loaded_samples.x0_values .+ pseudocount/1f6)
    xs = xs[:,x_perm]

    n = length(ts)
    num_test_samples = size(xs, 1)
    num_classes = size(qw_loc, 2)
    class_probs = zeros(Float64, num_classes)
    true_label_probs = zeros(Float64, (num_classes, n))

    # using TDist to avoid zero probabilities
    dist = TDist(20.0)
    sample_scales = estimate_sample_scales(xs, qx_bias_loc)
    xs .-= sample_scales

    # @show corspearman(qx_bias_loc, xs[1,:])
    # @show cor(qx_bias_loc, xs[1,:])

    # @show quantile(qx_bias_loc, 0.95)
    # for i in 1:num_test_samples
    #     @show (i, quantile(xs[i,:], 0.95), sum(exp.(xs[i,:])))
    # end
    # @show sample_scales
    # @show sum(exp.(qx_bias_loc))
    # exit()

    for i in 1:num_test_samples, j in 1:n
        for k in 1:num_classes
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

    # average across number of samples in each class
    class_counts = zeros(Int, num_classes)
    for idx in sample_class_idxs
        class_counts[idx] += 1
    end
    true_label_probs ./= class_counts

    return true_label_probs
end


"""
Evaluate classifier accuracy using kallisto bootstrap estimates.
"""
function evaluate_regression_with_kallisto_samples(
        testing_spec, ts, ts_metadata, x_perm, pseudocount,
        sample_class_idxs, qx_bias_loc, qw_loc, qx_scale)

    transcript_idx = Dict{String, Int}()
    for (j, t) in enumerate(ts)
        transcript_idx[t.metadata.name] = j
    end
    n = length(transcript_idx)
    num_classes = size(qw_loc, 2)
    class_probs = zeros(Float64, num_classes)
    true_label_probs = zeros(Float64, (num_classes, n))
    num_test_samples = length(testing_spec["samples"])

    filenames = [entry["kallisto"] for entry in testing_spec["samples"]]
    files = [h5open(filename) for filename in filenames]
    dist = TDist(20.0)

    num_efflens = length(files[1]["aux"]["eff_lengths"])
    efflens = zeros(Float32, (num_test_samples, num_efflens))

    for (i, file) in enumerate(files)
        for (j, efflen) in enumerate(read(file["aux"]["eff_lengths"]))
            efflens[i, j] = efflen
        end
    end

    transcript_ids = read(files[1]["aux"]["ids"])
    num_bootstrap_samples = length(files[1]["bootstrap"])

    xs = zeros(Float32, (num_test_samples, n))
    for boot_sample_num in 1:num_bootstrap_samples
        key = string("bs", boot_sample_num-1)
        for (i, file) in enumerate(files)
            counts = Vector{Float32}(read(file["bootstrap"][key]))
            props = PoleeModel.kallisto_counts_to_proportions(
                counts, (@view efflens[i,:]), pseudocount,
                transcript_ids, transcript_idx)
            props = props[1,x_perm]

            for k in 1:n
                xs[i,k] = log(props[k])
            end
        end

        sample_scales = estimate_sample_scales(xs, qx_bias_loc)
        xs .-= sample_scales

        for i in 1:num_test_samples, j in 1:n
            for k in 1:num_classes
                class_probs[k] = pdf(
                    dist,
                    (xs[i, j] - (qx_bias_loc[j] + qw_loc[j, k])) / qx_scale[j])
            end

            class_probs_sum = sum(class_probs)

            # true label probability
            true_label_probs[sample_class_idxs[i], j] +=
                (1/num_bootstrap_samples) *
                (class_probs_sum == 0.0 ? 1/num_classes :
                class_probs[sample_class_idxs[i]] / class_probs_sum)
        end
    end

    # average across number of samples in each class
    class_counts = zeros(Int, num_classes)
    for idx in sample_class_idxs
        class_counts[idx] += 1
    end
    true_label_probs ./= class_counts

    return true_label_probs
end


function main()
    parsed_args = parse_args(arg_settings)

    reg = CSV.read(parsed_args["regression-results"])

    ts, ts_metadata = load_transcripts_from_args(
        parsed_args, experiment_arg="experiment")

    testing_spec = YAML.load_file(parsed_args["experiment"])

    _, testing_sample_names, testing_sample_factors =
        PoleeModel.read_specification(testing_spec)

    num_test_samples = length(testing_sample_names)

    # Because we are reading parameters from a csv file, they are not necessarily
    # in the same order as ts, so we just put everything in sorted order by
    # transcript name. `x_perm` is permutation to put ts in sorted order.
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

    if parsed_args["point-estimates"] !== nothing && parsed_args["kallisto-bootstrap"]
        error("--point-estimates and --kallisto-bootstrap are mutually exclusive")
    end

    if parsed_args["point-estimates"] !== nothing
        true_label_probs = evaluate_regression_with_point_estimates(
            testing_spec, ts, ts_metadata, x_perm, parsed_args["point-estimates"],
            parsed_args["pseudocount"], sample_class_idxs, qx_bias_loc,
            qw_loc, qx_scale)
    elseif parsed_args["kallisto-bootstrap"]
        true_label_probs = evaluate_regression_with_kallisto_samples(
            testing_spec, ts, ts_metadata, x_perm, parsed_args["pseudocount"],
            sample_class_idxs, qx_bias_loc, qw_loc, qx_scale)
    else
        true_label_probs = evaluate_regression_with_likelihood_samples(
            testing_spec, ts, ts_metadata, x_perm, parsed_args["num-samples"],
            sample_class_idxs, qx_bias_loc, qw_loc, qx_scale)
    end

    dist = TDist(20.0)
    effect_size = log(abs(parsed_args["effect-size"]))
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
function draw_samples!(samplers, efflens, x_perm, xs)
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

