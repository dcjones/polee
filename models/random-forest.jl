
import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
import Polee

include(joinpath(dirname(pathof(Polee)), "PoleeModel.jl"))
using .PoleeModel

using ArgParse
using YAML
using Statistics
using Distributions
using StatsFuns
using Random
using DecisionTree
using DelimitedFiles
using SparseArrays
using LinearAlgebra: I


const arg_settings = ArgParseSettings()
arg_settings.prog = "polee model classify"
@add_arg_table arg_settings begin
    "--point-estimates"
        help = """
            Use point estimates read from a file specified in the experiment
            instead of approximated likelihood."""
        default = nothing
        arg_type = String
    "--genes"
        help = "Use gene expression rather than transcript expression."
        action = :store_true
    "--pseudocount"
        metavar = "C"
        help = "If specified with --point-estimates, add C tpm to each value."
        arg_type = Float64
    "--seed"
        help = "RNG seed used to partition data into test and training sets"
        default = 12345
        arg_type = Int
    "--annotations"
        help = "Transcript annotation file. If omitted, use h5 file to guess location."
        default = nothing
    "--ntrees"
        help = "Number of decision trees in the random forest."
        default = 200
        arg_type = Int
    "--nsubfeatures"
        help = """
            Number of predictors to used as a factor of the square root
            of the total number (of transcripts or genes).
            """
        default = 10.0
        arg_type = Float64
    "--partial-sampling"
        help = "Proportion of samples to subsample"
        default = 1.0
        arg_type = Float64
    "--training-samples"
        help = "Expand training data by sampling this many times from each sampler."
        default = 20
        arg_type = Int
    "--testing-samples"
        help = "Classify by averaging over this many samples."
        default = 20
        arg_type = Int
    "--output-predictions"
        help = "Output prediction probility matrix"
        default = "y-predicted.csv"
    "--output-truth"
        help = "Output true sample values for testing data"
        default = "y-true.csv"
    "training-experiment"
        metavar = "training.yml"
        help = "Training experiment specification"
        arg_type = String
    "testing-experiment"
        metavar = "testing.yml"
        help = "Testing xperiment specification"
        arg_type = String
    "factor"
        help = "Factors to regress on."
        arg_type = String
end


function main()
    parsed_args = parse_args(arg_settings)

    if parsed_args["annotations"] !== nothing
        ts, ts_metadata = Polee.Transcripts(parsed_args["annotations"])
    else
        ts, ts_metadata = load_transcripts_from_args(
            parsed_args, experiment_arg="training-experiment")
    end
    n = length(ts)

    features = I
    num_features = n
    if parsed_args["genes"]
        num_features, gene_idxs, transcript_idxs, gene_ids, gene_names =
            Polee.gene_feature_matrix(ts, ts_metadata)
        features = sparse(
            transcript_idxs, gene_idxs, ones(Float32, n), n, num_features)
    end

    training_spec = YAML.load_file(parsed_args["training-experiment"])
    testing_spec = YAML.load_file(parsed_args["testing-experiment"])

    Random.seed!(parsed_args["seed"])

    use_point_estimates = parsed_args["point-estimates"] !== nothing

    if use_point_estimates
        training_loaded_samples = load_point_estimates_from_specification(
            training_spec, ts, ts_metadata, parsed_args["point-estimates"])
        testing_loaded_samples = load_point_estimates_from_specification(
            testing_spec, ts, ts_metadata, parsed_args["point-estimates"])

        if parsed_args["pseudocount"] !== nothing
            training_loaded_samples.x0_values .+= parsed_args["pseudocount"] ./ 1f6
            testing_loaded_samples.x0_values .+= parsed_args["pseudocount"] ./ 1f6
        end

        x_training = log.(training_loaded_samples.x0_values * features)
        x_testing = log.(testing_loaded_samples.x0_values * features)

        training_sample_names = training_loaded_samples.sample_names
        training_sample_factors = training_loaded_samples.sample_factors

        testing_sample_names = testing_loaded_samples.sample_names
        testing_sample_factors = testing_loaded_samples.sample_factors
    else
        x_samplers_training, training_efflens, training_sample_names, training_sample_factors =
                load_samplers_from_specification(training_spec, ts, ts_metadata)

        x_samplers_testing, testing_efflens, testing_sample_names, testing_sample_factors =
                load_samplers_from_specification(testing_spec, ts, ts_metadata)
    end

    num_training_samples = length(training_sample_names)
    num_testing_samples = length(testing_sample_names)

    y_true_onehot_training, factor_idx = build_factor_matrix(
        num_training_samples, training_sample_factors, parsed_args["factor"])

    y_true_onehot_testing, _ = build_factor_matrix(
        num_testing_samples, testing_sample_factors, parsed_args["factor"],
        factor_idx)

    factor_names = collect(keys(factor_idx))
    num_factors = length(factor_idx)

    decode_onehot(y) = Int[idx[2] for idx in argmax(y, dims=2)[:,1]]
    y_true_training = decode_onehot(y_true_onehot_training)
    y_true_testing = decode_onehot(y_true_onehot_testing)

    ntrees = parsed_args["ntrees"]
    nsubfeatures = min(n, round(Int, parsed_args["nsubfeatures"] * sqrt(n)))

    if use_point_estimates
        println("training...")
        forest = build_forest(
            y_true_training, x_training, nsubfeatures, ntrees,
            parsed_args["partial-sampling"])

        println("testing...")
        y_predicted = apply_forest_proba(
            forest, x_testing, collect(1:num_factors))
    else
        # Build an enlarged training set by drawing samples from the
        # approximated likelihood
        println("training...")
        x_training = Array{Float32}(undef,
            (num_training_samples * parsed_args["training-samples"], n))
        draw_samples!(x_samplers_training, training_efflens, x_training)
        x_training *= features
        y_true_training_expanded = repeat(y_true_training, parsed_args["training-samples"])

        forest = build_forest(
            y_true_training_expanded, x_training, nsubfeatures,
            ntrees, parsed_args["partial-sampling"])

        # Evaluate by drawing samples and averaging the result
        println("testing...")
        x_test = similar(testing_efflens)
        x_test_features = Array{Float32}(undef, (size(x_test, 1), num_features))
        y_predicted = zeros(Float64, (num_testing_samples, num_factors))
        for i in 1:parsed_args["testing-samples"]
            draw_samples!(x_samplers_testing, testing_efflens, x_test)
            x_test_features .= x_test * features
            y_predicted .+= apply_forest_proba(
                forest, x_test_features, collect(1:num_factors))
        end
        y_predicted ./= parsed_args["testing-samples"]
    end

    write_classification_probs(
        factor_names,
        parsed_args["output-predictions"],
        parsed_args["output-truth"],
        y_predicted,
        y_true_onehot_testing)
end


"""
Draw sample from approximated likelihood, adjust for effective length and log
transform.
"""
function draw_samples!(samplers, efflens, xs)
    num_samples = length(samplers)
    # conceivably this could draw from the same sampler twice in different threads
    # which would break things. Sampling isn't a big bottleneck anyway.
    # Threads.@threads for i in 1:size(xs, 1)
    for i in 1:size(xs, 1)
        xs_row = @view xs[i,:]
        k = mod1(i, num_samples)
        rand!(samplers[k], xs_row)
        xs_row ./= @view efflens[k,:]
        xs_row ./= sum(xs_row)
        map!(log, xs_row, xs_row)
    end
end


function build_factor_matrix(num_samples, sample_factors, factor, factor_idx=nothing)
    factor_options = Set{Union{Missing, String}}()

    for sample_factors in sample_factors
        push!(factor_options, string(get(sample_factors, factor, missing)))
    end

    # assign indexes to factors
    if factor_idx === nothing
        nextidx = 1
        factor_names = String[]
        factor_idx = Dict{String, Int}()
        for option in factor_options
            factor_idx[option] = nextidx
            push!(factor_names, string(factor, ":", option))
            nextidx += 1
        end
    end

    num_factors = length(factor_idx)
    F = zeros(Float32, (num_samples, num_factors))

    for (i, sample_factors) in enumerate(sample_factors)
        option = get(sample_factors, factor, missing)
        if haskey(factor_idx, option)
            F[i, factor_idx[option]] = 1
        end
    end

    return F, factor_idx
end


function write_classification_probs(
        factor_names, y_predicted_filename, y_true_filename,
         y_predicted, y_true)

    open(y_true_filename, "w") do output
        print(output, factor_names[1])
        for i in 2:length(factor_names)
            print(output, ",", factor_names[i])
        end
        println(output)

        for i in 1:size(y_true, 1)
            print(output, y_true[i, 1])
            for j in 2:size(y_true, 2)
                print(output, ",", y_true[i, j])
            end
            println(output)
        end
    end

    open(y_predicted_filename, "w") do output
        print(output, factor_names[1])
        for i in 2:length(factor_names)
            print(output, ",", factor_names[i])
        end
        println(output)

        for i in 1:size(y_predicted, 1)
            print(output, y_predicted[i, 1])
            for j in 2:size(y_predicted, 2)
                print(output, ",", y_predicted[i, j])
            end
            println(output)
        end
    end
end


main()
