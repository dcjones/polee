
import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
import Polee

include(joinpath(dirname(pathof(Polee)), "PoleeModel.jl"))
using .PoleeModel

using ArgParse
using YAML
using PyCall
import Random


const arg_settings = ArgParseSettings()
arg_settings.prog = "polee model pca"
@add_arg_table arg_settings begin
    "--feature"
        metavar = "F"
        action = :store_arg
        default = "transcript"
        help = "One of transcript, gene, splicing"
    "--num-components"
        metavar = "N"
        help = "Number of PCA components"
        default = 2
        arg_type = Int
    "--posterior-mean"
        help = "Use posterior mean point estimate instead of the full model"
        action = :store_true
        default = false
    "--output-z"
        metavar = "filename"
        help = "Output file for PCA projection"
        default = "pca-z.csv"
    "--output-w"
        metavar = "filename"
        help = "Output file for PCA transcript weights"
        default = nothing
    "--max-num-samples"
        metavar = "N"
        help = "Only run the model on a randomly selected subset of N samples"
        default = nothing
        arg_type = Int
    "experiment"
        metavar = "experiment.yml"
        help = "Experiment specification"
end


function main()
    parsed_args = parse_args(arg_settings)

    feature = parsed_args["feature"]

    if feature ∉ ["transcript", "gene", "splicing"]
        error(string(parsed_args["feature"], " is not a supported feature."))
    end

    if feature ∈ ["gene"]
        error(string(parsed_args["feature"], " feature is not yet implemented."))
    end

    ts, ts_metadata = load_transcripts_from_args(parsed_args)

    init_python_modules()
    polee_pca_py = pyimport("polee_pca")
    tf = pyimport("tensorflow")

    # so we get the same subset when max-num-samples is used
    Random.seed!(1234)

    loaded_samples = load_samples_from_specification(
        YAML.load_file(parsed_args["experiment"]),
        ts, ts_metadata,
        max_num_samples=parsed_args["max-num-samples"])

    num_samples, n = size(loaded_samples.x0_values)
    num_pca_components = Int(get(parsed_args, "num-components", 2))
    x0_log = log.(loaded_samples.x0_values)

    if feature == "transcript"
        z, w = polee_pca_py.estimate_transcript_pca(
            loaded_samples.init_feed_dict, num_samples, n,
            loaded_samples.variables, x0_log, num_pca_components,
            parsed_args["posterior-mean"])

        if parsed_args["output-w"] !== nothing
            write_pca_w(parsed_args["output-w"], ts, w)
        end
    elseif feature == "splicing"
        gene_db = write_transcripts("genes.db", ts, ts_metadata)
        sess = tf.Session()
        qx_loc, qx_scale = approximate_splicing_likelihood(
            loaded_samples, ts, ts_metadata, gene_db, sess)

        # free up some memory
        tf.reset_default_graph()
        sess.close()
        # create_tensorflow_variables!(loaded_samples)
        num_features = size(qx_loc, 2)

        z, w = polee_pca_py.estimate_feature_pca(
            qx_loc, qx_scale, num_samples, num_features, num_pca_components,
            parsed_args["posterior-mean"])

        if parsed_args["output-w"] !== nothing
            write_pca_w(
                parsed_args["output-w"], "splicing_event",
                String[string(i) for i in 1:num_features], w)
        end
    end

    if parsed_args["output-z"] !== nothing
        write_pca_z(parsed_args["output-z"], loaded_samples.sample_names, z)
    end
end


function write_pca_w(output_filename, feature_type, feature_names, w)
    n, num_components = size(w)
    open(output_filename, "w") do output
        print(output, feature_type)
        for j in 1:num_components
            print(output, ",component", j)
        end
        println(output)

        for (i, feature_name) in enumerate(feature_names)
            print(output, feature_name)
            for j in 1:num_components
                print(output, ",", w[i, j])
            end
            println(output)
        end
    end
end


function write_pca_w(output_filename, ts, w)
    write_pca_w(
        output_filename, "transcript_id",
        String[t.metadata.name for t in ts], w)
end


function write_pca_z(output_filename, sample_names, z)
    num_samples, num_pca_components = size(z)
    open(output_filename, "w") do output
        print(output, "sample")
        for j in 1:num_pca_components
            print(output, ",component", j)
        end
        println(output)

        for (i, sample) in enumerate(sample_names)
            print(output, sample)
            for j in 1:num_pca_components
                print(output, ",", z[i, j])
            end
            println(output)
        end
    end
end

main()

