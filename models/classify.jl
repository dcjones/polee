
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
    polee_regression_py = pyimport("polee_regression")
    polee_py = pyimport("polee")
    tf = pyimport("tensorflow")

    training_spec = YAML.load_file(parsed_args["training-experiment"])
    testing_spec = YAML.load_file(parsed_args["testing-experiment"])

    use_point_estimates = parsed_args["point-estimates"] !== nothing
    pseudocount = parsed_args["pseudocount"] === nothing ? 0.0 : parsed_args["pseudocount"]

    if use_point_estimates
        training_loaded_samples = load_point_estimates_from_specification(
             training_spec, ts, ts_metadata, parsed_args["point-estimates"])
        testing_loaded_samples = load_point_estimates_from_specification(
             training_spec, ts, ts_metadata, parsed_args["point-estimates"])

        if parsed_args["pseudocount"] !== nothing
            loaded_samples.x0_values .+= parsed_args["pseudocount"] / 1f6
        end
    else
        training_loaded_samples = load_samples_from_specification(
            training_spec, ts, ts_metadata)
        testing_loaded_samples = load_samples_from_specification(
            training_spec, ts, ts_metadata)

        if parsed_args["pseudocount"] !== nothing
            error("--pseudocount argument only valid with --point-estimates")
        end
    end

    num_training_samples = length(training_loaded_samples.sample_factors)
    num_testing_samples = length(testing_loaded_samples.sample_factors)

    y_true_onehot_training, factor_idx = build_factor_matrix(
        num_training_samples, training_loaded_samples.sample_factors, parsed_args["factor"])

    y_true_onehot_testing, _ = build_factor_matrix(
        num_testing_samples, testing_loaded_samples.sample_factors, parsed_args["factor"],
        factor_idx)

    factor_names = collect(keys(factor_idx))
    num_factors = length(factor_idx)

    decode_onehot(y) = Int[idx[2] for idx in argmax(y, dims=2)[:,1]]
    y_true_training = decode_onehot(y_true_onehot_training)
    y_true_testing = decode_onehot(y_true_onehot_testing)

    if feature == "gene"
        num_features, gene_idxs, transcript_idxs, gene_ids, gene_names =
            Polee.gene_feature_matrix(ts, ts_metadata)

        p = sortperm(transcript_idxs)
        permute!(gene_idxs, p)
        permute!(transcript_idxs, p)

        gene_sizes = zeros(Float32, num_features)
        for i in gene_idxs
            gene_sizes[i] += 1
        end

        training_num_samples = length(training_loaded_samples.sample_factors)
        training_x_gene_init, training_x_isoform_init = gene_initial_values(
            gene_idxs, transcript_idxs,
            training_loaded_samples.x0_values, training_num_samples, num_features, n)

        training_sample_scales = estimate_sample_scales(
            log.(training_loaded_samples.x0_values), upper_quantile=0.95)

        regression = polee_regression_py.RNASeqGeneLinearRegression(
            training_loaded_samples.variables,
            gene_idxs, transcript_idxs,
            training_x_gene_init, training_x_isoform_init,
            gene_sizes,
            y_true_onehot_training,
            training_sample_scales,
            use_point_estimates)
        regression.fit(10000)

        testing_num_samples = length(training_loaded_samples.sample_factors)
        testing_x_gene_init, testing_x_isoform_init = gene_initial_values(
            gene_idxs, transcript_idxs,
            testing_loaded_samples.x0_values, testing_num_samples, num_features, n)

        testing_sample_scales = estimate_sample_scales(
            log.(testing_loaded_samples.x0_values), upper_quantile=0.95)

        y_predicted = regression.classify(
            testing_loaded_samples.variables,
            gene_idxs, transcript_idxs,
            testing_x_gene_init, testing_x_isoform_init,
            gene_sizes, testing_sample_scales, use_point_estimates,
            5000)
    else
        training_x_init = log.(training_loaded_samples.x0_values)
        training_sample_scales = estimate_sample_scales(training_x_init)
        regression = polee_regression_py.RNASeqTranscriptLinearRegression(
                training_loaded_samples.variables,
                training_x_init,
                y_true_onehot_training,
                training_sample_scales,
                use_point_estimates)
        regression.fit(6000)

        testing_x_init = log.(testing_loaded_samples.x0_values)
        testing_sample_scales = estimate_sample_scales(testing_x_init)

        y_predicted = regression.classify(
            testing_loaded_samples.variables,
            testing_x_init,
            testing_sample_scales,
            use_point_estimates,
            3000)
        @show y_predicted
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

