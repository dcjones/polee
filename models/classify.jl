
import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
import Polee

include(joinpath(dirname(pathof(Polee)), "PoleeModel.jl"))
using .PoleeModel

using ArgParse
using YAML
using PyCall
using Statistics
using Distributions
using StatsFuns
using Random
using DecisionTree
using LinearAlgebra: I
using HDF5

include("kallisto.jl")

const arg_settings = ArgParseSettings()
arg_settings.prog = "polee model classify"
@add_arg_table arg_settings begin
    "--feature"
        metavar = "F"
        action = :store_arg
        default = "transcript"
        help = "One of transcript, gene, splicing"
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
    "--kallisto"
        help = """
            Use kallisto maximum likelihood estimates. The sample
            specifications should have a `kallisto` key pointing to the h5
            file.
            """
        action = :store_true
    "--pseudocount"
        metavar = "C"
        help = "If specified with --point-estimates, add C tpm to each value."
        arg_type = Float64
    "--annotations"
        help = "Transcript annotation file. If omitted, use h5 file to guess location."
        default = nothing
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
    feature = parsed_args["feature"]

    if parsed_args["annotations"] !== nothing
        ts, ts_metadata = Polee.Transcripts(parsed_args["annotations"])
    else
        ts, ts_metadata = load_transcripts_from_args(
            parsed_args, experiment_arg="training-experiment")
    end
    n = length(ts)

    init_python_modules()
    polee_classify_py = pyimport("polee_classify")
    polee_py = pyimport("polee")
    tf = pyimport("tensorflow")

    training_spec = YAML.load_file(parsed_args["training-experiment"])
    testing_spec = YAML.load_file(parsed_args["testing-experiment"])

    use_point_estimates = parsed_args["point-estimates"] !== nothing
    pseudocount = parsed_args["pseudocount"] === nothing ? 0.0 : parsed_args["pseudocount"]

    if parsed_args["kallisto"] && parsed_args["kallisto-bootstrap"]
        error("Only one of '--kallisto' and '--kallisto-bootstrap' can be used.")
    end

    if (parsed_args["kallisto"] || parsed_args["kallisto-bootstrap"])
        if use_point_estimates
            error("'--use-point-estimates' in not compatible with '--kallisto' or '--kallisto-bootstrap'")
        end

        _, training_sample_names, training_sample_factors =
            PoleeModel.read_specification(training_spec)
        _, testing_sample_names, testing_sample_factors =
            PoleeModel.read_specification(testing_spec)
    end

    features = I

    if parsed_args["kallisto"]
        x_training = read_kallisto_estimates(training_spec, pseudocount, features)
        x_testing = read_kallisto_estimates(testing_spec, pseudocount, features)
        num_features = size(x_training, 2)
        use_point_estimates = true

    elseif parsed_args["kallisto-bootstrap"]
        x_training_bootstraps = read_kallisto_bootstrap_samples(training_spec, pseudocount, features)
        x_testing_bootstraps = read_kallisto_bootstrap_samples(testing_spec, pseudocount, features)
        num_features = size(x_training_bootstraps[1], 2)

    elseif use_point_estimates
        training_loaded_samples = load_point_estimates_from_specification(
             training_spec, ts, ts_metadata, parsed_args["point-estimates"])
        testing_loaded_samples = load_point_estimates_from_specification(
             testing_spec, ts, ts_metadata, parsed_args["point-estimates"])

        if parsed_args["pseudocount"] !== nothing
            training_loaded_samples.x0_values .+= parsed_args["pseudocount"] ./ 1f6
            testing_loaded_samples.x0_values .+= parsed_args["pseudocount"] ./ 1f6
        end

        x_training = log.(training_loaded_samples.x0_values)
        x_testing = log.(testing_loaded_samples.x0_values)

        training_sample_names = training_loaded_samples.sample_names
        training_sample_factors = training_loaded_samples.sample_factors

        testing_sample_names = testing_loaded_samples.sample_names
        testing_sample_factors = testing_loaded_samples.sample_factors
    else
        training_loaded_samples = load_samples_from_specification(
            training_spec, ts, ts_metadata)
        testing_loaded_samples = load_samples_from_specification(
            testing_spec, ts, ts_metadata)

        training_sample_names = training_loaded_samples.sample_names
        training_sample_factors = training_loaded_samples.sample_factors

        testing_sample_names = testing_loaded_samples.sample_names
        testing_sample_factors = testing_loaded_samples.sample_factors

        if parsed_args["pseudocount"] !== nothing
            error("--pseudocount argument only valid with --point-estimates")
        end
    end

    num_training_samples = length(training_spec["samples"])
    num_testing_samples = length(testing_spec["samples"])

    y_true_onehot_training, factor_idx = build_factor_matrix(
        num_training_samples, training_sample_factors, parsed_args["factor"])

    y_true_onehot_testing, _ = build_factor_matrix(
        num_testing_samples, testing_sample_factors, parsed_args["factor"],
        factor_idx)

    num_factors = length(factor_idx)
    factor_names = Vector{String}(undef, num_factors)
    for (k, v) in factor_idx
        factor_names[v] = k
    end

    decode_onehot(y) = Int[idx[2] for idx in argmax(y, dims=2)[:,1]]
    y_true_training = decode_onehot(y_true_onehot_training)
    y_true_testing = decode_onehot(y_true_onehot_testing)

    if feature == "gene"
        error("Error gene classification not implemented.")
    else
        # TODO: handle kallisto-bootstrap

        if use_point_estimates
            n = size(x_training, 2)
            classifier = polee_classify_py.RNASeqLogisticRegression(num_factors, n)

            classifier.fit(
                x_training,
                y_true_onehot_training, 5000)

            y_predicted = classifier.predict(x_testing)
        else
            classifier = polee_classify_py.RNASeqLogisticRegression(num_factors, n)

            classifier.fit_sample(
                num_training_samples, n, training_loaded_samples.variables,
                y_true_onehot_training, 5000)

            y_predicted = classifier.predict_sample(
                num_testing_samples, n, testing_loaded_samples.variables, 100)
        end
    end

    write_classification_probs(
        factor_names,
        parsed_args["output-predictions"],
        parsed_args["output-truth"],
        y_predicted,
        y_true_onehot_testing)
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

