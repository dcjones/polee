
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

const arg_settings = ArgParseSettings()
arg_settings.prog = "polee model cluster"
@add_arg_table arg_settings begin
    # "--max-num-components"
    #     metavar = "N"
    #     help = "Number of PCA components"
    #     default = 30
    #     arg_type = Int
    "--training-prop"
        help = "Proportion of samples to use for training."
        default = 0.25
        arg_type = Float64
    "--max-num-samples"
        metavar = "N"
        help = "Only run the model on a randomly selected subset of N samples"
        default = nothing
        arg_type = Int
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

    init_python_modules()
    polee_regression_py = pyimport("polee_regression")
    tf = pyimport("tensorflow")

    # so we get the same subset when max-num-samples is used
    Random.seed!(1234)

    loaded_samples = load_samples_from_specification(
        YAML.load_file(parsed_args["experiment"]),
        ts, ts_metadata,
        max_num_samples=parsed_args["max-num-samples"],
        using_tensorflow=false)

    num_samples, n = size(loaded_samples.x0_values)

    z_true, factor_names =
        build_factor_matrix(loaded_samples, parsed_args["factor"])

    x0 = log.(loaded_samples.x0_values)

    sess = tf.Session()
    polee_classify_py = pyimport("polee_classify")

    num_train_samples = round(Int, num_samples*parsed_args["training-prop"])
    idx = shuffle(1:num_samples)
    train_idx = idx[1:num_train_samples]
    test_idx = idx[num_train_samples+1:end]

    sess = tf.Session()

    # I don't actually have to do batches here. Everything should just be
    # initialization.

    create_tensorflow_variables!(loaded_samples, num_train_samples)

    feed_dict_subset = subset_feed_dict(loaded_samples.init_feed_dict, train_idx)

    # TODO: need to subset everything that we feed in here
    classify_model = polee_classify_py.train_classifier(
        sess,
        feed_dict_subset,
        num_train_samples,
        n,
        loaded_samples.variables,
        x0[train_idx,:],
        z_true[train_idx,:])

    # TODO: now run a separate test function. Need to reset the graph/sess in
    # between.

    # sess = tf.Session()
    # num_test_samples = num_samples - num_train_samples
    # create_tensorflow_variables!(loaded_samples, num_test_samples)

    # TODO: function to make confusion matrix. 

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
    # TODO:
end


main()

