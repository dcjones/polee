
import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
import Polee

include(joinpath(dirname(pathof(Polee)), "PoleeModel.jl"))
using .PoleeModel

using ArgParse
using YAML
using PyCall


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

    if parsed_args["feature"] ∉ ["transcript", "gene", "splicing"]
        error(string(parsed_args["feature"], " is not a supported feature."))
    end

    if parsed_args["feature"] ∈ ["gene", "splicing"]
        error(string(parsed_args["feature"], " feature is not yet implemented."))
    end

    ts, ts_metadata = load_transcripts_from_args(parsed_args)

    init_python_modules()
    polee_pca_py = pyimport("polee_pca")

    loaded_samples = load_samples_from_specification(
        YAML.load_file(parsed_args["experiment"]),
        ts, ts_metadata,
        max_num_samples=parsed_args["max-num-samples"])

    num_samples, n = size(loaded_samples.x0_values)
    num_pca_components = Int(get(parsed_args, "num-components", 2))
    x0_log = log.(loaded_samples.x0_values)
    z, w = polee_pca_py.estimate_transcript_pca(
        loaded_samples.init_feed_dict, num_samples, n,
        loaded_samples.variables, x0_log, num_pca_components)

    if parsed_args["output-w"] !== nothing
        write_pca_w(parsed_args["output-w"], ts, w)
    end

    if parsed_args["output-z"] !== nothing
        write_pca_z(parsed_args["output-z"], loaded_samples.sample_names, z)
    end
end


function write_pca_w(output_filename, ts, w)
    n, num_components = size(w)
    open(output_filename, "w") do output
        print(output, "transcript_id")
        for j in 1:num_components
            print(output, ",component", j)
        end
        println(output)

        for (i, t) in enumerate(ts)
            @assert i == t.metadata.id
            print(output, t.metadata.name)
            for j in 1:num_components
                print(output, ",", w[i, j])
            end
            println(output)
        end
    end
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

