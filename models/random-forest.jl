
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
using DelimitedFiles


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
    n = length(ts)
    feature_names = String[t.metadata.name for t in ts]

    spec = YAML.load_file(parsed_args["experiment"])

    if parsed_args["point-estimates"] !== nothing
        Random.seed!(parsed_args["seed"])
        loaded_samples = load_point_estimates_from_specification(
            spec, ts, ts_metadata, parsed_args["point-estimates"],
            max_num_samples=parsed_args["max-num-samples"])

        if parsed_args["pseudocount"] !== nothing
            loaded_samples.x0_values .+= parsed_args["pseudocount"] / 1f6
        end

        x0 = log.(loaded_samples.x0_values)
        num_samples = size(x0, 1)

        sample_names = loaded_samples.sample_names
        sample_factors = loaded_samples.sample_factors

        # x_mean = mean(x0, dims=1)[1,:]
        # p = sortperm(x_mean, rev=true)
        # @show p[1:10]
        # @show x0[:,p[1]]
        # @show x0[:,p[2]]


        # @show sum(loaded_samples.x0_values, dims=2)
        # @show size(x0)
        # @show typeof(x0)
        # exit()
    else
        Random.seed!(parsed_args["seed"])
        samplers, efflens, sample_names, sample_factors = load_samplers_from_specification(
            spec, ts, ts_metadata,
            max_num_samples=parsed_args["max-num-samples"])

        # TODO: this should be combined into one function that loads tf variables
        # and builds samplers

        init_python_modules()
        Random.seed!(parsed_args["seed"])
        loaded_samples = load_samples_from_specification(
            spec, ts, ts_metadata,
            max_num_samples=parsed_args["max-num-samples"])
        x0 = log.(loaded_samples.x0_values)
        num_samples = size(x0, 1)

        sample_names = loaded_samples.sample_names
        sample_factors = loaded_samples.sample_factors

        efflens = loaded_samples.efflen_values

        # x_mean = mean(x0, dims=1)[1,:]
        # p = sortperm(x_mean, rev=true)
        # @show p[1:10]
        # @show x0[:,p[1]]
        # @show x0[:,p[2]]

        # @show sum(loaded_samples.x0_values, dims=2)
        # @show size(x0)
        # @show typeof(x0)
        # exit()

        polee_regression_py = pyimport("polee_regression")
        tf = pyimport("tensorflow")
    end

    z_true_onehot, factor_names = build_factor_matrix(
        sample_names, sample_factors, parsed_args["factor"])
    num_factors = size(z_true_onehot, 2)
    K = length(factor_names)

    use_point_estimates = parsed_args["point-estimates"] !== nothing

    # sample_scales = estimate_sample_scales(x0, upper_quantile=0.9)

    num_train_samples = round(Int, num_samples*parsed_args["training-prop"])
    idx = shuffle(1:num_samples)
    train_idx = idx[1:num_train_samples]
    test_idx = idx[num_train_samples+1:end]

    decode_onehot(y) = Int[idx[2] for idx in argmax(y, dims=2)[:,1]]
    z_true = decode_onehot(z_true_onehot)

    n_trees = 400

    if use_point_estimates
        forest = build_forest(z_true[train_idx], x0[train_idx,:], -1, n_trees)
        z_predict = apply_forest_proba(forest, x0[test_idx,:], collect(1:num_factors))
    else
        empty!(loaded_samples.variables)
        empty!(loaded_samples.init_feed_dict)
        tf.reset_default_graph()
        create_tensorflow_variables!(loaded_samples, num_train_samples)

        train_feed_dict_subset = subset_feed_dict(
            loaded_samples.init_feed_dict, train_idx)

        sample_scales = estimate_sample_scales(x0[train_idx,:], upper_quantile=0.9)
        fill!(sample_scales, 0.0f0)

        sess = tf.Session()

        qx_loc = x0[train_idx,:]
        # qx_loc, qw_loc, qw_scale, qx_bias, qx_scale, =
        #     polee_regression_py.estimate_transcript_linear_regression(
        #         train_feed_dict_subset, loaded_samples.variables,
        #         x0[train_idx,:],
        #         # zeros(Float32, (num_train_samples, 0)),
        #         z_true_onehot[train_idx,:],
        #         sample_scales,
        #         parsed_args["point-estimates"], sess,
        #         # 4000)
        #         800)

        # open("train-x.csv", "w") do output
        #     writedlm(output, qx_loc, ",")
        # end

        # open("train-y.csv", "w") do output
        #     train_y = z_true[train_idx,:]
        #     writedlm(output, train_y, ",")
        # end

        # TODO: I could also try building the forest by sampling from the
        # posterion for x


        forest = build_forest(
            z_true[train_idx], qx_loc, -1, n_trees)

        empty!(loaded_samples.variables)
        empty!(loaded_samples.init_feed_dict)
        tf.reset_default_graph()
        sess.close()
        sess = tf.Session()

        num_test_samples = length(test_idx)
        create_tensorflow_variables!(loaded_samples, num_test_samples)

        test_feed_dict_subset = subset_feed_dict(
            loaded_samples.init_feed_dict, test_idx)

        sample_scales = estimate_sample_scales(x0[test_idx,:], upper_quantile=0.9)
        fill!(sample_scales, 0.0f0)

        # qx_loc, qw_loc, qw_scale, qx_bias, qx_scale, =
        #     polee_regression_py.estimate_transcript_linear_regression(
        #         test_feed_dict_subset, loaded_samples.variables,
        #         x0[test_idx,:], zeros(Float32, (num_test_samples, 0)), sample_scales,
        #         parsed_args["point-estimates"], sess, 8000)

        # open("test-x.csv", "w") do output
        #     writedlm(output, qx_loc, ",")
        # end

        # open("test-y.csv", "w") do output
        #     test_y = z_true[test_idx,:]
        #     writedlm(output, test_y, ",")
        # end



        # forest = build_forest_with_samples(
        #     z_true[train_idx], samplers[train_idx], efflens[train_idx,:], -1, n_trees)



        samplers_test = samplers[test_idx]
        efflens_test = efflens[test_idx,:]

        xs = similar(efflens_test)
        xs_row = Array{Float32}(undef, n)

        # n_predict_iter = 400
        # fill!(xs, 0.0)
        # for i in 1:n_predict_iter
        #     @show i

        #     # draw new sample
        #     for j in 1:length(samplers_test)
        #         rand!(samplers_test[j], xs_row)
        #         xs_row ./= efflens_test[j,:]
        #         xs_row ./= sum(xs_row)
        #         xs[j,:] .+= xs_row
        #     end
        # end
        # xs ./= n_predict_iter
        # z_predict = apply_forest_proba(
        #     forest, xs, collect(1:num_factors))

        z_predict = zeros(Float64, (length(test_idx), num_factors))
        n_predict_iter = 200
        for i in 1:n_predict_iter
            @show i

            # draw new sample
            for j in 1:length(samplers_test)
                rand!(samplers_test[j], xs_row)
                xs[j,:] = xs_row
            end
            xs ./= efflens_test
            xs ./= sum(xs, dims=2)

            z_predict .+= apply_forest_proba(
                forest, log.(xs), collect(1:num_factors))
        end
        z_predict ./= n_predict_iter

        # @show typeof(qx_loc)
        # @show size(qx_loc)
        # z_predict = apply_forest_proba(
        #     forest, qx_loc, collect(1:num_factors))
    end

    write_classification_probs(factor_names, z_predict, z_true_onehot[test_idx,:])
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


function build_factor_matrix(sample_names, sample_factors, factor)
    factor_options = Set{Union{Missing, String}}()

    for sample_factors in sample_factors
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

    num_samples = length(sample_names)
    num_factors = length(factor_idx)
    F = zeros(Float32, (num_samples, num_factors))

    for (i, sample_factors) in enumerate(sample_factors)
        option = get(sample_factors, factor, missing)
        if haskey(factor_idx, option)
            F[i, factor_idx[option]] = 1
        end
    end

    return F, factor_names
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


function build_forest_with_samples(
        labels              :: Vector{T},
        samplers            :: Vector{Polee.ApproxLikelihoodSampler},
        efflens             :: Array{Float32, 2},
        n_subfeatures       = -1,
        n_trees             = 10,
        partial_sampling    = 0.7,
        max_depth           = -1,
        min_samples_leaf    = 1,
        min_samples_split   = 2,
        min_purity_increase = 0.0;
        rng                 = Random.GLOBAL_RNG) where {S, T}

    if n_trees < 1
        throw("the number of trees must be >= 1")
    end
    if !(0.0 < partial_sampling <= 1.0)
        throw("partial_sampling must be in the range (0,1]")
    end

    if n_subfeatures == -1
        n_features = size(efflens, 2)
        n_subfeatures = round(Int, sqrt(n_features))
    end

    t_samples = length(labels)
    n_samples = floor(Int, partial_sampling * t_samples)

    num_samples = length(samplers)
    n = size(efflens, 2)
    features = similar(efflens)
    feature_row = Array{Float32}(undef, n)


    # manually doing posterior mean
    # fill!(features, 0.0f0)
    # for i in 1:n_trees
    #     for j in 1:length(samplers)
    #         rand!(samplers[j], feature_row)
    #         feature_row ./= efflens[j,:]
    #         feature_row ./= sum(feature_row)
    #         features[j,:] .+= feature_row
    #     end
    # end
    # features ./= n_trees


    # rngs = mk_rng(rng)::Random.AbstractRNG
    rngs = rng
    forest = DecisionTree.LeafOrNode{Float32,Int}[]
    for i in 1:n_trees
        @show i
        inds = rand(rngs, 1:t_samples, n_samples)

        # draw new sample
        for j in 1:length(samplers)
            rand!(samplers[j], feature_row)
            features[j,:] = feature_row
        end
        features ./= efflens
        features ./= sum(features, dims=2)
        features = log.(features)

        tree = build_tree(
            labels[inds],
            features[inds,:],
            n_subfeatures,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            min_purity_increase,
            rng = rngs)
        push!(forest, tree)
    end

    return Ensemble{Float32, Int}(forest)
end

main()