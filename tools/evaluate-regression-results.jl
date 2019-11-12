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
    "--output-binned"
        metavar = "filename"
        default = "regression-evaluation-binned.csv"
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
    "--prob-bin-size"
        default = 0.05
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
    # map!(sqrt, qw_scale, qw_scale) # std. dev. to variance

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

    class_probs_row = zeros(Float64, num_classes)

    scales = ones(Float64, num_test_samples)

    # TODO: New plan: what we really want to do is to evaluate the cumuluative
    # accuracy of all the de calls within a probability bin in order to get
    # a plot that rewards making confident calls.

    # That means we have to do binning in this function.


    # compute binned DE probabilities
    dist = TDist(10.0)
    effect_size = log(abs(parsed_args["effect-size"]))
    prob_bin_size = parsed_args["prob-bin-size"]
    de_prob_bins = zeros(Int, (num_classes, n))
    for i in 1:num_classes, j in 1:n
        prob_down = cdf(dist, (-effect_size - qw_loc[j,i]) / qw_scale[j,i])
        prob_up = ccdf(dist, (effect_size - qw_loc[j,i]) / qw_scale[j,i])
        prob_de = prob_down + prob_up

        de_prob_bins[i, j] = round(Int, prob_de / prob_bin_size) + 1
    end
    num_prob_bins = round(Int, 1.0 / prob_bin_size) + 1
    class_probs = zeros(Float64, (num_test_samples, n, num_classes))
    binned_class_probs = zeros(Float64, (num_prob_bins, num_classes))

    predictions = zeros(Float64, num_prob_bins)
    bin_counts = zeros(Int, num_prob_bins)

    entropy = zeros(Float64, (num_classes, n))
    binned_entropy = zeros(Float64, (num_classes, num_prob_bins))


    # TODO: maybe I should compute average class probabilities
    # then compute everything else on that expectation?

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
                # TODO: maybe this underestimates the probability because Int
                # does not consider x variance.
                # I wonder if qw_scale should be used at all here...

                # class_probs[i, j, k] = pdf(Normal(
                #     qx_bias_loc[j] + qw_loc[j, k],
                #     qx_scale[j]^2),
                #     xs[i, j])

                class_probs[i, j, k] = pdf(
                    dist,
                    (xs[i, j] - (qx_bias_loc[j] + qw_loc[j, k])) / qx_scale[j])
            end

            class_probs_sum = sum(class_probs[i, j, :])
            entropy[sample_class_idxs[i], j] +=
                class_probs_sum == 0.0 ? 1/num_classes :
                class_probs[sample_class_idxs[i]] / class_probs_sum
        end

        # TODO: Now I'm thinking we should build a classifier by
        # doing what we were doing:
        #   use TDist to avoid zero probs
        #   compute product (log-sum) of probabilities in each bin for each sample
        #   take the argmax
        #   estimate cumulative accuracy
        #
        # This is tricky with multi-class, because two tissues may have
        # nearly the same DE genes, so they can be distinguished from everything
        # else but not from each other, thus producing low predictive performance.
        #
        # Hopefully that averages out or something.

        for i in 1:num_test_samples
            true_k = sample_class_idxs[i]

            fill!(binned_class_probs, 0.0)
            for j in 1:n, k in 1:num_classes
                bin = de_prob_bins[true_k, j]
                binned_class_probs[bin, k] += log(class_probs[i, j, k])
            end

            fill!(predictions, 0.0)
            for bin in 1:num_prob_bins
                predictions[bin] += argmax(binned_class_probs[bin,:]) == true_k
            end

            @show (true_k, predictions[1])

            binned_entropy[true_k, :] .+= predictions
        end
    end

    # average across sampler draws
    entropy ./= parsed_args["num-samples"]
    binned_entropy ./= parsed_args["num-samples"]

    # average across number of samples in each class
    class_counts = zeros(Int, num_classes)
    for idx in sample_class_idxs
        class_counts[idx] += 1
    end
    entropy ./= class_counts
    binned_entropy ./= class_counts

    open(parsed_args["output"], "w") do output
        println(output, "transcript_id,factor,de_prob,classification_entropy")
        for i in 1:num_classes, j in 1:n
            prob_down = cdf(dist, (-effect_size - qw_loc[j,i]) / qw_scale[j,i])
            prob_up = ccdf(dist, (effect_size - qw_loc[j,i]) / qw_scale[j,i])
            prob_de = prob_down + prob_up

            println(
                output,
                tnames[j], ",",
                factor_names[i], ",",
                prob_de, ",",
                entropy[i, j])
        end
    end

    open(parsed_args["output-binned"], "w") do output
        println(output, "transcript_id,factor,classification_entropy")
        for i in 1:num_classes, j in 1:num_prob_bins

            println(
                output,
                factor_names[i], ",",
                prob_bin_size * (j-1), ",",
                binned_entropy[i, j])
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

