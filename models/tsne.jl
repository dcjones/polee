
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
arg_settings.prog = "polee model tsne"
@add_arg_table arg_settings begin
    "--feature"
        metavar = "F"
        action = :store_arg
        default = "transcript"
        help = "One of transcript, gene, splicing"
    "--num-components"
        metavar = "N"
        help = "Number of t-SNE dimensions"
        default = 2
        arg_type = Int
    "--posterior-mean"
        help = "Use posterior mean point estimate instead of the full model"
        action = :store_true
        default = false
    "--output-z"
        metavar = "filename"
        help = "Output file for t-SNE projection"
        default = "tsne-z.csv"
    "--max-num-samples"
        metavar = "N"
        help = "Only run the model on a randomly selected subset of N samples"
        default = nothing
        arg_type = Int
    "--batch-size"
        metavar = "N"
        help = "Sample over mini-batches of size N"
        default = 100
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

    if feature ∈ ["splicing", "gene"]
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
        # batch_size=parsed_args["batch-size"])

    num_samples, n = size(loaded_samples.x0_values)
    batch_size = min(parsed_args["batch-size"], num_samples)
    num_pca_components = Int(get(parsed_args, "num-components", 2))
    x0_log = log.(loaded_samples.x0_values)

    sess = tf.Session()
    polee_tsne_py = pyimport("polee_tsne")

    z = polee_tsne_py.estimate_tsne(
        loaded_samples.init_feed_dict,
        loaded_samples.variables,
        log.(loaded_samples.x0_values),
        parsed_args["num-components"],
        batch_size, false, sess,
        use_vlr=true)

    if parsed_args["output-z"] !== nothing
        write_pca_z(parsed_args["output-z"], loaded_samples.sample_names, z)
    end
end


function find_sigmas(x0, target_perplexity, use_vlr::Bool)
    num_samples, n = size(x0)
    σs = Array{Float32}(undef, num_samples)
    rowdiff = Array{Float32}(undef, n)
    δ = Array{Float32}(undef, num_samples)
    δσ = Array{Float32}(undef, num_samples)
    for i in 1:num_samples

        # compute distance
        for j in 1:num_samples
            for k in 1:n
                rowdiff[k] = x0[i,k] - x0[j,k]
            end

            if use_vlr
                δ[j] = var(rowdiff)
            else
                δ[j] = 0.0f0
                for k in 1:n
                    δ[j] += rowdiff[k]^2
                end
            end
        end

        σ_lower = 1e-2
        σ_upper = 10.0 * sqrt(maximum(δ))

        perplexity = Inf
        while abs(target_perplexity - perplexity) > 0.1
            σ = (σ_lower + σ_upper) / 2
            δσ[i] = 0.0
            for j in 1:num_samples
                if j != i
                    δσ[j] = exp(-δ[j] / (2*σ^2))
                end
            end
            δσ_sum = sum(δσ)

            if δσ_sum == 0.0
                sigma_lower = σ
                continue
            end

            H = 0.0
            for j in 1:num_samples
                if j != i
                    pji = δσ[j] / δσ_sum
                    if pji > 1e-16
                        H -= pji * log2(pji)
                    end
                end
            end
            perplexity = 2.0^H
            @show (i,perplexity,target_perplexity)

            if perplexity > target_perplexity
                σ_lower = σ
            else
                σ_upper = σ
            end
        end

        σs[i] = σ
    end

    return σs
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