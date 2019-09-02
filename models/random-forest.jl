
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
    "--seed"
        help = "RNG seed used to partition data into test and training sets"
        default = 12345
        arg_type = Int
    "--ntrees"
        help = "Number of decision trees in the random forest."
        default = 400
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

    ts, ts_metadata = load_transcripts_from_args(
        parsed_args, experiment_arg="training-experiment")
    n = length(ts)
    feature_names = String[t.metadata.name for t in ts]

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
            training_loaded_samples.x0_values .+= parsed_args["pseudocount"] / 1f6
            testing_loaded_samples.x0_values .+= parsed_args["pseudocount"] / 1f6
        end

        x_training = log.(training_loaded_samples.x0_values)
        x_testing = log.(testing_loaded_samples.x0_values)

        training_sample_names = training_loaded_samples.sample_names
        training_sample_factors = training_loaded_samples.sample_factors

        testing_sample_names = testing_loaded_samples.sample_names
        testing_sample_factors = testing_loaded_samples.sample_factors
    else
        init_python_modules()
        training_loaded_samples = load_samples_from_specification(
            training_spec, ts, ts_metadata)

        training_sample_names = training_loaded_samples.sample_names
        training_sample_factors = training_loaded_samples.sample_factors

        polee_regression_py = pyimport("polee_regression")
        tf = pyimport("tensorflow")

        x_samplers_testing, testing_efflens, testing_sample_names,
            testing_sample_factors =
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

    if use_point_estimates
        forest = build_forest(y_true_training, x_training, -1, ntrees)

        y_predicted = apply_forest_proba(
            forest, x_testing, collect(1:num_factors))
    else
        sess = tf.Session()

        # Let's think about this.
        #
        #

        sample_scales = estimate_sample_scales(
            log.(training_loaded_samples.x0_values), upper_quantile=0.9)
        fill!(sample_scales, 0.0f0)

        qx_loc_training, _, _, _, _ =
            polee_regression_py.estimate_transcript_linear_regression(
                training_loaded_samples.init_feed_dict,
                training_loaded_samples.variables,
                log.(training_loaded_samples.x0_values),
                y_true_onehot_training,
                # zeros(Float32, (num_training_samples, 1)),
                sample_scales,
                use_point_estimates, sess,
                800)

        # The shrinkage model may very well shift the scales so that qx_loc_training
        # doesn't sum to one
        @show log.(sum(exp.(qx_loc_training), dims=2))
        qx_loc_training .-= log.(sum(exp.(qx_loc_training), dims=2))

        sess.close()

        forest = build_forest(
           y_true_training, qx_loc_training, -1, ntrees)

        xs = similar(testing_efflens)
        xs_row = Array{Float32}(undef, n)
        y_predicted = zeros(Float64, (num_testing_samples, num_factors))
        n_predict_iter = 200
        for i in 1:n_predict_iter
            # draw new sample
            for j in 1:num_testing_samples
                rand!(x_samplers_testing[j], xs_row)
                xs[j,:] = xs_row
            end
            xs ./= testing_efflens
            xs ./= sum(xs, dims=2)

            sample_scales = estimate_sample_scales(log.(xs), upper_quantile=0.9)
            fill!(sample_scales, 0.0f0)

            y_predicted .+= apply_forest_proba(
                forest, log.(xs) .- sample_scales, collect(1:num_factors))
        end
        y_predicted ./= n_predict_iter
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