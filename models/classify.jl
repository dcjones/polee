
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
    # "--max-num-components"
    #     metavar = "N"
    #     help = "Number of PCA components"
    #     default = 30
    #     arg_type = Int
    "--training-prop"
        help = "Proportion of samples to use for training."
        default = 0.75
        arg_type = Float64
    "--max-num-samples"
        metavar = "N"
        help = "Only run the model on a randomly selected subset of N samples"
        default = nothing
        arg_type = Int
    "--point-estimates"
        help = """
            Use point estimates read from a file specified in the experiment
            instead of approximated likelihood."""
        default = nothing
        arg_type = String
    "--discriminative"
        help = "Train a alternative discriminative model instead of the default generative one"
        action = :store_true
    "--pseudocount"
        metavar = "C"
        help = "If specified with --point-estimates, add C tpm to each value."
        arg_type = Float64
    "--seed"
        help = "RNG seed used to partition data into test and training sets"
        default = 12345
        arg_type = Int
    "--output"
        metavar = "filename"
        help = "Output file for confusion matrix"
        default = "confusion-matrix.csv"
    "experiment"
        metavar = "experiment.yml"
        help = "Experiment specification"
        arg_type = String
    "factor"
        help = "Factors to regress on."
        arg_type = String
end



function main()
    parsed_args = parse_args(arg_settings)

    ts, ts_metadata = load_transcripts_from_args(parsed_args)
    feature_names = String[t.metadata.name for t in ts]

    init_python_modules()
    polee_regression_py = pyimport("polee_regression")
    tf = pyimport("tensorflow")

    # so we get the same subset when max-num-samples is used
    Random.seed!(parsed_args["seed"])

    spec = YAML.load_file(parsed_args["experiment"])

    if parsed_args["point-estimates"] !== nothing
        loaded_samples = load_point_estimates_from_specification(
            spec, ts, ts_metadata, parsed_args["point-estimates"],
            max_num_samples=parsed_args["max-num-samples"])

        if parsed_args["pseudocount"] !== nothing
            loaded_samples.x0_values .+= parsed_args["pseudocount"] / 1f6
        end
    else
        loaded_samples = load_samples_from_specification(
            spec, ts, ts_metadata,
            max_num_samples=parsed_args["max-num-samples"],
            # using_tensorflow=false)
            using_tensorflow=true)

        if parsed_args["pseudocount"] !== nothing
            error("--pseudocount argument only valid with --point-estimates")
        end
    end

    num_samples, n = size(loaded_samples.x0_values)

    z_true, factor_names =
        build_factor_matrix(loaded_samples, parsed_args["factor"])
    @show factor_names
    K = length(factor_names)

    x0 = log.(loaded_samples.x0_values)


    use_point_estimates = parsed_args["point-estimates"] !== nothing

    # @show exp.(estimate_sample_scales(x0, upper_quantile=0.85))
    # @show exp.(estimate_sample_scales(x0, upper_quantile=0.9))
    # @show exp.(estimate_sample_scales(x0, upper_quantile=0.95))
    # @show exp.(estimate_sample_scales(x0, upper_quantile=0.99))
    # exit()

    sample_scales = estimate_sample_scales(x0, upper_quantile=0.9)
    # fill!(sample_scales, 0.0)

    polee_regression_py = pyimport("polee_regression")
    polee_classify_py = pyimport("polee_classify")

    sess = tf.Session()



    # Let's try estimating expression then using those as point estimates

    # TODO:
    # This is a promising approach, but could be considered cheating to use
    # the full data to produce shrinkage estimates. Consider parititioning into
    # as separately producing point estimates.

    # Also, if I'm going to be generating point estimates, I may as well use
    # some ML library so I can try out different models.

    # qx_loc, qw_loc, qw_scale, qx_bias, qx_scale, =
    #     polee_regression_py.estimate_transcript_linear_regression(
    #         loaded_samples.init_feed_dict, loaded_samples.variables,
    #         x0, zeros(Float32, (num_samples, 0)), sample_scales, parsed_args["point-estimates"], sess)

    # x0 = qx_loc

    # use_point_estimates = true

    empty!(loaded_samples.variables)
    empty!(loaded_samples.init_feed_dict)
    tf.reset_default_graph()
    sess.close()
    sess = tf.Session()

    num_train_samples = round(Int, num_samples*parsed_args["training-prop"])
    idx = shuffle(1:num_samples)
    train_idx = idx[1:num_train_samples]
    test_idx = idx[num_train_samples+1:end]


    # TODO: write data to files so we can independently tinker with it
    # open("train-x.csv", "w") do output
    #     train_x = x0[train_idx,:]
    #     writedlm(output, train_x, ",")
    # end

    # open("test-x.csv", "w") do output
    #     test_x = x0[test_idx,:]
    #     writedlm(output, test_x, ",")
    # end

    # open("train-y.csv", "w") do output
    #     train_y = z_true[train_idx,:]
    #     writedlm(output, train_y, ",")
    # end

    # open("test-y.csv", "w") do output
    #     test_y = z_true[test_idx,:]
    #     writedlm(output, test_y, ",")
    # end



    # I don't actually have to do batches here. Everything should just be
    # initialization.

    if !use_point_estimates
        create_tensorflow_variables!(loaded_samples, num_train_samples)
    end

    train_feed_dict_subset = subset_feed_dict(
        loaded_samples.init_feed_dict, train_idx)

    # [num_samples, num_factors]
    @show sum(z_true[train_idx,:], dims=1)
    @show sum(z_true[test_idx,:], dims=1)

    @show z_true[test_idx,:]

    if parsed_args["discriminative"]
        classify_model = polee_classify_py.train_classifier(
            sess,
            train_feed_dict_subset,
            num_train_samples,
            n,
            loaded_samples.variables,
            x0[train_idx,:],
            z_true[train_idx,:],
            sample_scales[train_idx],
            use_point_estimates)

        write_weights(classify_model["lyrn"], feature_names)
    else
        classify_model = polee_classify_py.train_probabalistic_classifier(
            sess,
            train_feed_dict_subset,
            num_train_samples,
            n,
            loaded_samples.variables,
            x0[train_idx,:],
            z_true[train_idx,:],
            sample_scales[train_idx],
            use_point_estimates)
    end

    tf.reset_default_graph()
    sess.close()
    sess = tf.Session()
    tf.keras.backend.set_session(sess)


    num_test_samples = num_samples - num_train_samples

    if !use_point_estimates
        create_tensorflow_variables!(loaded_samples, num_test_samples)
    end

    test_feed_dict_subset = subset_feed_dict(
        loaded_samples.init_feed_dict, test_idx)

    if parsed_args["discriminative"]
        z_predict = polee_classify_py.run_classifier(
            sess,
            classify_model,
            test_feed_dict_subset,
            num_test_samples,
            n,
            loaded_samples.variables,
            x0[test_idx,:],
            K,
            sample_scales[test_idx],
            use_point_estimates)
    else
        z_predict, w, x_bias = polee_classify_py.run_probabalistic_classifier(
            sess,
            classify_model,
            test_feed_dict_subset,
            num_test_samples,
            n,
            loaded_samples.variables,
            x0[test_idx,:],
            K,
            sample_scales[test_idx],
            use_point_estimates)
    end

    # TODO: Inspecting model
    z_true_test = z_true[test_idx,:]
    idx_a = Int[]
    idx_b = Int[]
    for i in 1:size(z_true_test, 1)
        if argmax(z_true_test[i,:]) == 1
            push!(idx_a, i)
        else
            push!(idx_b, i)
        end
    end

    @show extrema(x0)
    @show quantile(x0[1,:], [0.0, 0.1, 0.5, 0.9, 1.0])

    # println("regression examples")
    # p1 = sortperm(abs.(w[:,1]), rev=true)
    # p2 = sortperm(abs.(w[:,2]), rev=true)
    # for i in 1:5
    #     j = p1[i]
    #     @show (i, j, w[j,1], w[j,2])
    #     @show x_bias[j]
    #     @show round.(x0[test_idx,j][idx_a])
    #     @show round.(x0[test_idx,j][idx_b])
    #     println("--------------------")
    # end

    # for i in 1:5
    #     j = p2[i]
    #     @show (i, p2[i], w[j,1], w[j,2])
    #     @show x_bias[j]
    #     @show round.(x0[test_idx,j][idx_a])
    #     @show round.(x0[test_idx,j][idx_b])
    #     println("--------------------")
    # end

    # # TODO: Let's try to manually find examples
    # println("manual examples")
    # x0_a_mean = mean(x0[test_idx,:][idx_a,:], dims=1)[1,:]
    # x0_b_mean = mean(x0[test_idx,:][idx_b,:], dims=1)[1,:]

    # p = sortperm(abs.(x0_a_mean .- x0_b_mean),rev=true)
    # for i in 1:10
    #     j = p[i]
    #     @show (i, j, w[j,1], w[j,2])
    #     @show x_bias[j]
    #     @show round.(x0[test_idx,j][idx_a])
    #     @show round.(x0[test_idx,j][idx_b])
    #     println("--------------------")
    # end

    @show z_predict

    M = confusion_matrx(z_predict, z_true[test_idx,:])

    write_confusion_matrix(parsed_args["output"], factor_names, M)


    # Testing on training data just to debug

    # create_tensorflow_variables!(loaded_samples, num_train_samples)

    # train_feed_dict_subset = subset_feed_dict(
    #     loaded_samples.init_feed_dict, train_idx)

    # z_predict = polee_classify_py.run_classifier(
    #     sess,
    #     classify_model,
    #     train_feed_dict_subset,
    #     num_train_samples,
    #     n,
    #     loaded_samples.variables,
    #     x0[train_idx,:],
    #     K)

    # M = confusion_matrx(z_predict, z_true[train_idx,:])

    true_count = 0
    for i in 1:size(M, 1)
        true_count += M[i, i]
    end
    true_rate = true_count / sum(M)

    num_samples = length(loaded_samples.sample_names)

    # TODO: For debugging
    open("classify-results.csv", "a") do output
        println(
            output,
            use_point_estimates ? "point" : "full",
            ",",
            parsed_args["seed"],
            ",",
            num_samples,
            ",",
            parsed_args["training-prop"],
            ",",
            true_rate)
    end

    # TODO: also for debugging
    write_classification_probs(factor_names, z_predict, z_true[test_idx,:])
end


function write_classification_probs(factor_names, z_predict, z_true)
    open("z_true.csv", "w") do output
        print(output, factor_names[1])
        for i in 2:length(factor_names)
            print(output, ",", factor_names[i])
        end
        println(output)

        for i in 1:size(z_true, 1)
            print(output, z_true[i, 1])
            for j in 2:size(z_true, 2)
                print(output, ",", z_true[i, j])
            end
            println(output)
        end
    end

    open("z_predict.csv", "w") do output
        print(output, factor_names[1])
        for i in 2:length(factor_names)
            print(output, ",", factor_names[i])
        end
        println(output)

        for i in 1:size(z_predict, 1)
            print(output, z_predict[i, 1])
            for j in 2:size(z_predict, 2)
                print(output, ",", z_predict[i, j])
            end
            println(output)
        end
    end
end


function write_confusion_matrix(filename, factor_names, M)
    open(filename, "w") do output
        println(output, "true_factor,predicted_factor,count")
        for i in 1:size(M, 1), j in 1:size(M, 2)
            println(
                output,
                factor_names[i], ",",
                factor_names[j], ",",
                M[i,j])
        end
    end
end


function subset_feed_dict(feed_dict, idx)
    feed_dict_subset = Dict()
    for (k, v) in feed_dict
        if length(size(v)) > 1
            feed_dict_subset[k] = v[idx,:]
        else
            feed_dict_subset[k] = v[idx]
        end
    end
    return feed_dict_subset
end


function build_factor_matrix(loaded_samples, factor)
    factor_options = Set{Union{Missing, String}}()

    for sample_factors in loaded_samples.sample_factors
        push!(factor_options, string(get(sample_factors, factor, missing)))
    end

    # assign indexes to factors
    nextidx = 1
    factor_names = String[]
    factor_idx = Dict{String, Int}()
    for option in factor_options
        factor_idx[option] = nextidx
        push!(factor_names, string(factor, ":", option))
        nextidx += 1
    end

    num_samples = length(loaded_samples.sample_names)
    num_factors = length(factor_idx)
    F = zeros(Float32, (num_samples, num_factors))

    for (i, sample_factors) in enumerate(loaded_samples.sample_factors)
        option = get(sample_factors, factor, missing)
        if haskey(factor_idx, option)
            F[i, factor_idx[option]] = 1
        end
    end

    return F, factor_names
end


function confusion_matrx(z_predict, z_true)
    @assert size(z_predict) == size(z_true)

    num_samples, num_classes = size(z_true)

    # ML confusion matrix
    # M = zeros(Int, (num_classes, num_classes))
    # for (i, j) in zip(argmax(z_true, dims=2), argmax(z_predict, dims=2))
    #     M[i[2], j[2]] += 1
    # end

    # (num_samples, num_facors)

    # expected confusion matrix
    M = zeros(Float64, (num_classes, num_classes))
    for i in 1:num_samples
        j = argmax(z_true[i,:])
        for k in 1:num_classes
            M[j, k] += z_predict[i, k]
        end
    end

    return M
end


function write_weights(ws, feature_names)
    @show ws[2]
    open("nn-weights.csv", "w") do output
        println(output, "transcript_id,b,w")
        for i in 1:size(ws[1], 1)
            println(output, feature_names[i], ",", ws[1][i, 1], ",", ws[1][i, 2])
        end
    end
end


main()

