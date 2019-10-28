
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


const arg_settings = ArgParseSettings()
arg_settings.prog = "polee model imputation"
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
    "--confounders"
        help = "Comma separated list of confounding factors"
        default = nothing
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
    polee_imputation_py = pyimport("polee_imputation")
    polee_py = pyimport("polee")
    tf = pyimport("tensorflow")

    training_spec = YAML.load_file(parsed_args["training-experiment"])
    testing_spec = YAML.load_file(parsed_args["testing-experiment"])

    spec = deepcopy(training_spec)
    append!(spec["samples"], testing_spec["samples"])

    use_point_estimates = parsed_args["point-estimates"] !== nothing
    pseudocount = parsed_args["pseudocount"] === nothing ? 0.0 : parsed_args["pseudocount"]

    if use_point_estimates
        loaded_samples = load_point_estimates_from_specification(
             spec, ts, ts_metadata, parsed_args["point-estimates"])

        if parsed_args["pseudocount"] !== nothing
            loaded_samples.x0_values .+= parsed_args["pseudocount"] / 1f6
        end
    else
        loaded_samples = load_samples_from_specification(
            spec, ts, ts_metadata)

        if parsed_args["pseudocount"] !== nothing
            error("--pseudocount argument only valid with --point-estimates")
        end
    end

    num_training_samples = length(training_spec["samples"])
    num_testing_samples = length(testing_spec["samples"])
    num_samples = num_training_samples + num_testing_samples
    @assert length(loaded_samples.sample_names) == num_training_samples + num_testing_samples

    y_true_onehot, factor_idx = build_factor_matrix(
        num_samples, loaded_samples.sample_factors, parsed_args["factor"])

    num_factors = length(factor_idx)
    factor_names = Vector{String}(undef, num_factors)
    for (k, v) in factor_idx
        factor_names[v] = k
    end

    confounders = nothing
    if parsed_args["confounders"] !== nothing
        confounders, _ = build_factor_matrix(
            num_samples, loaded_samples.sample_factors,
            split(parsed_args["confounders"], ','))
        num_factors += size(confounders, 2)
    end

    y_true_onehot_training = y_true_onehot[1:num_training_samples, :]
    y_true_onehot_testing = y_true_onehot[num_training_samples+1:end, :]

    decode_onehot(y) = Int[idx[2] for idx in argmax(y, dims=2)[:,1]]

    if feature == "gene"
        error("Gene regression not implementated")
    else
        x_init = log.(loaded_samples.x0_values)
        sample_scales = estimate_sample_scales(x_init)

        regression = polee_imputation_py.RNASeqImputedTranscriptLinearRegression(
                loaded_samples.variables,
                x_init,
                y_true_onehot_training,
                confounders,
                sample_scales,
                use_point_estimates)
        # y_predicted = regression.fit(5000 + 25*num_samples)
        y_predicted = regression.fit(6000)
    end

    write_classification_probs(
        factor_names,
        parsed_args["output-predictions"],
        parsed_args["output-truth"],
        y_predicted,
        y_true_onehot_testing)
end

function build_factor_matrix(num_samples, sample_factors, factor::String, factor_idx=nothing)
    return build_factor_matrix(num_samples, sample_factors, [factor], factor_idx)
end

function build_factor_matrix(num_samples, sample_factors, factors::Vector, factor_idx=nothing)

    factor_options = Set{Union{Missing, String}}()
    for sample_factors in sample_factors
        for factor in factors
            option = string(factor, ":", get(sample_factors, factor, missing))
            push!(factor_options, option)
        end
    end

    # assign indexes to factors
    if factor_idx === nothing
        nextidx = 1
        factor_names = String[]
        factor_idx = Dict{String, Int}()
        for option in factor_options
            factor_idx[option] = nextidx
            push!(factor_names, option)
            nextidx += 1
        end
    end

    num_factors = length(factor_idx)
    F = zeros(Float32, (num_samples, num_factors))

    for (i, sample_factors) in enumerate(sample_factors)
        for factor in factors
            option = string(factor, ":", get(sample_factors, factor, missing))
            if haskey(factor_idx, option)
                F[i, factor_idx[option]] = 1
            end
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
