
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
    "--nonredundant"
        help = "Avoid overparameterization by excluding one factor from each group"
        action = :store_true
        default = false
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
    # polee_imputation_py = pyimport("polee_imputation")
    polee_regression_py = pyimport("polee_regression")

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
        # loaded_samples = load_samples_from_specification(
        #     spec, ts, ts_metadata)

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
    num_samples = num_training_samples + num_testing_samples
    # @assert length(loaded_samples.sample_names) == num_training_samples + num_testing_samples

    # y_true_onehot, factor_idx = build_factor_matrix(
    #     num_samples, loaded_samples.sample_factors, parsed_args["factor"])

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

    num_factors = length(factor_idx)
    factor_names = Vector{String}(undef, num_factors)
    for (k, v) in factor_idx
        factor_names[v] = k
    end

    confounders = nothing
    # if parsed_args["confounders"] !== nothing
    #     confounders, _ = build_factor_matrix(
    #         num_samples, loaded_samples.sample_factors,
    #         split(parsed_args["confounders"], ','))
    #     num_factors += size(confounders, 2)
    # end

    # y_true_onehot_training = y_true_onehot[1:num_training_samples, :]
    # y_true_onehot_testing = y_true_onehot[num_training_samples+1:end, :]

    decode_onehot(y) = Int[idx[2] for idx in argmax(y, dims=2)[:,1]]

    if feature == "gene"
        error("Gene regression not implementated")
    else
        # x_init = log.(loaded_samples.x0_values)
        # sample_scales = estimate_sample_scales(x_init)

        tnames = [t.metadata.name for t in ts]

        # open("x_init.csv", "w") do output
        #     print(output, "transcript_id")
        #     for i in 1:size(x_init, 1)
        #         print(output, ",sample", i)
        #     end
        #     println(output)
        #     for j in 1:size(x_init, 2)
        #         print(output, tnames[j])
        #         for i in 1:size(x_init, 1)
        #         print(output, ",", x_init[i, j])
        #         end
        #         println(output)
        #     end
        # end

        # regression = polee_imputation_py.RNASeqImputedTranscriptLinearRegression(
        #         loaded_samples.variables,
        #         x_init,
        #         y_true_onehot_training,
        #         # 0, # confounders
        #         confounders,
        #         sample_scales,
        #         use_point_estimates)
        # # y_predicted = regression.fit(500 + 25*num_samples)
        # y_predicted, qw_loc_var, x_loc, qx_scale_loc_var, qx_loc_var = regression.fit(1000)

        test_x_init = log.(testing_loaded_samples.x0_values)
        test_sample_scales = estimate_sample_scales(test_x_init)

        train_x_init = log.(training_loaded_samples.x0_values)
        train_sample_scales = estimate_sample_scales(train_x_init)

        regression = polee_regression_py.RNASeqTranscriptLinearRegression(
                training_loaded_samples.variables,
                train_x_init,
                y_true_onehot_training,
                train_sample_scales,
                use_point_estimates)

        regression.fit(600)

        y_predicted = regression.classify(
            testing_loaded_samples.variables,
            test_x_init,
            test_sample_scales,
            use_point_estimates,
            400)

        # open("x.csv", "w") do output
        #     print(output, "transcript_id")
        #     for i in 1:size(qx_loc_var, 1)
        #         print(output, ",sample", i)
        #     end
        #     println(output)
        #     for j in 1:size(qx_loc_var, 2)
        #         print(output, tnames[j])
        #         for i in 1:size(qx_loc_var, 1)
        #         print(output, ",", qx_loc_var[i, j])
        #         end
        #         println(output)
        #     end
        # end

        # open("qw_loc.csv", "w") do output
        #     print(output, "transcript_id")
        #     for j in 1:length(factor_names)
        #         print(output, ",", factor_names[j])
        #     end
        #     for j in length(factor_names)+1:size(qw_loc_var, 1)
        #         print(output, ",", "extra", j-length(factor_names))
        #     end
        #     println(output)
        #     for i in 1:size(qw_loc_var, 2)
        #         print(output, tnames[i])
        #         for j in 1:size(qw_loc_var, 1)
        #             print(output, ",", qw_loc_var[j, i])
        #         end
        #         println(output)
        #     end
        # end

        # open("x_loc.csv", "w") do output
        #     print(output, "transcript_id")
        #     for j in 1:length(factor_names)
        #         print(output, ",", factor_names[j])
        #     end
        #     for j in length(factor_names)+1:size(x_loc, 1)
        #         print(output, ",", "extra", j-length(factor_names))
        #     end
        #     println(output)
        #     for i in 1:size(x_loc, 2)
        #         print(output, tnames[i])
        #         for j in 1:size(x_loc, 1)
        #             print(output, ",", x_loc[j, i])
        #         end
        #         println(output)
        #     end
        # end

        # open("qx_scale_loc.csv", "w") do output
        #     println(output, "transcript_id,scale")
        #     for i in 1:length(tnames)
        #         println(output, tnames[i], ",", qx_scale_loc_var[i])
        #     end
        # end

        # p = sortperm(abs.(qw_loc_var[end,:]), rev=true)
        # for i in 1:10
        #     println(tnames[p[i]])
        #     @show x_init[:,p[i]]
        # end
    end

    write_classification_probs(
        factor_names,
        parsed_args["output-predictions"],
        parsed_args["output-truth"],
        y_predicted,
        y_true_onehot_testing)
end

# function build_factor_matrix(num_samples, sample_factors, factor::String,
#         nonredundant::Bool=false, factor_idx=nothing)
#     return build_factor_matrix(num_samples, sample_factors, [factor], nonredundant, factor_idx)
# end

# function build_factor_matrix(num_samples, sample_factors, factors::Vector,
#         nonredundant::Bool=false, factor_idx=nothing)

#     factor_options = Set{Union{Missing, String}}()
#     for sample_factors in sample_factors
#         for factor in factors
#             option = string(factor, ":", get(sample_factors, factor, missing))
#             push!(factor_options, option)
#         end
#     end

#     # remove one factor from each group to make them non-redundant
#     if nonredundant
#         for k in keys(factor_options)
#             if missing ∈ factor_options[k]
#                 delete!(factor_options[k], missing)
#             else
#                 delete!(factor_options[k], first(factor_options[k]))
#             end
#         end
#     end

#     # assign indexes to factors
#     if factor_idx === nothing
#         nextidx = 1
#         factor_names = String[]
#         factor_idx = Dict{String, Int}()
#         for option in factor_options
#             factor_idx[option] = nextidx
#             push!(factor_names, option)
#             nextidx += 1
#         end
#     end

#     num_factors = length(factor_idx)
#     F = zeros(Float32, (num_samples, num_factors))

#     for (i, sample_factors) in enumerate(sample_factors)
#         for factor in factors
#             option = string(factor, ":", get(sample_factors, factor, missing))
#             if haskey(factor_idx, option)
#                 F[i, factor_idx[option]] = 1
#             end
#         end
#     end

#     return F, factor_idx
# end

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
