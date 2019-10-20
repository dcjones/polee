
import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
import Polee

include(joinpath(dirname(pathof(Polee)), "PoleeModel.jl"))
using .PoleeModel

using ArgParse
using YAML
using PyCall
using Statistics
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
    "--output-z"
        metavar = "filename"
        help = "Output file for PCA projection"
        default = "pca-z.csv"
    "--output-w"
        metavar = "filename"
        help = "Output file for PCA transcript weights"
        default = nothing
    # "--neural-network"
    #     help = """
    #     Use a neural network in place of the linear transformation.
    #     (i.e. Probabalistic decoder instead of PCA)
    #     """
    #     action = :store_true
    "experiment"
        metavar = "experiment.yml"
        help = "Experiment specification"
end


function main()
    parsed_args = parse_args(arg_settings)


    ts, ts_metadata = load_transcripts_from_args(parsed_args)
    n = length(ts)

    init_python_modules()
    use_point_estimates = parsed_args["point-estimates"] !== nothing
    spec = YAML.load_file(parsed_args["experiment"])
    polee_pca_py = pyimport("polee_pca")

    if parsed_args["point-estimates"] !== nothing
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

    num_samples, n = size(loaded_samples.x0_values)
    num_pca_components = Int(get(parsed_args, "num-components", 2))

    x0_log = log.(loaded_samples.x0_values)

    sample_scales = estimate_sample_scales(x0_log, upper_quantile=0.9)

    pca = polee_pca_py.RNASeqPCA(
        loaded_samples.variables, x0_log, sample_scales, use_point_estimates)

    z, w = pca.fit(6000)

    if parsed_args["output-w"] !== nothing
        write_pca_w(parsed_args["output-w"], ts, w)
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

