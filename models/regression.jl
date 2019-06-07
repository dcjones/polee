
import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
import Polee

include(joinpath(dirname(pathof(Polee)), "PoleeModel.jl"))
using .PoleeModel

using ArgParse
using YAML
using PyCall
using Statistics
using Distributions
using Printf: @printf
import Random


const arg_settings = ArgParseSettings()
arg_settings.prog = "polee model pca"
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
    "--output"
        metavar = "filename"
        help = "Output file for regression coefficients"
        default = "regression-coefficients.csv"
    "--lower-credible"
        metavar = "L"
        default = 0.025
        arg_type = Float64
    "--upper-credible"
        metavar = "U"
        default = 0.975
        arg_type = Float64
    "--max-num-samples"
        metavar = "N"
        help = "Only run the model on a randomly selected subset of N samples"
        default = nothing
        arg_type = Int
    "--factors"
        help = """
            Comma-separated list of factors to regress on. (Default: use all factors)
        """
        default = nothing
        arg_type = String
    "--nonredundant"
        help = "Avoid overparameterization by excluding one factor from each group"
        action = :store_true
        default = false
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

    if feature ∈ ["gene", "splicing"]
        error(string(parsed_args["feature"], " feature is not yet implemented."))
    end

    ts, ts_metadata = load_transcripts_from_args(parsed_args)

    init_python_modules()
    polee_regression_py = pyimport("polee_regression")
    tf = pyimport("tensorflow")

    # so we get the same subset when max-num-samples is used
    Random.seed!(1234)

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
            max_num_samples=parsed_args["max-num-samples"])

        if parsed_args["pseudocount"] !== nothing
            error("--pseudocount argument only valid with --point-estimates")
        end
    end

    num_samples, n = size(loaded_samples.x0_values)
    x0_log = log.(loaded_samples.x0_values)

    q0 = parsed_args["lower-credible"]
    q1 = parsed_args["upper-credible"]

    factors = parsed_args["factors"] === nothing ?
        nothing : split(parsed_args["factors"], ',')

    factor_matrix, factor_names = build_factor_matrix(
        loaded_samples, factors, parsed_args["nonredundant"])

    qw_loc, qw_scale, qx_scale =
        polee_regression_py.estimate_transcript_linear_regression(
            loaded_samples.init_feed_dict, loaded_samples.variables,
            x0_log, factor_matrix, parsed_args["point-estimates"])

    write_regression_effects(
        parsed_args["output"],
        factor_names,
        String[t.metadata.name for t in ts],
        qw_loc, qw_scale,
        parsed_args["lower-credible"],
        parsed_args["upper-credible"])
end


function build_factor_matrix(
        loaded_samples, factors, nonredundant::Bool=false)
    # figure out possibilities for each factor
    if factors === nothing
        factors_set = Set{String}()
        for sample_factors in loaded_samples.sample_factors
            for factor in keys(sample_factors)
                push!(factors_set, factor)
            end
        end
        factors = collect(String, factors_set)
    end

    factor_options = Dict{String, Set{Union{Missing, String}}}()
    for factor in factors
        factor_options[factor] = Set{Union{Missing, String}}()
    end

    for sample_factors in loaded_samples.sample_factors
        for factor in factors
            push!(
                factor_options[factor],
                string(get(sample_factors, factor, missing)))
        end
    end

    # remove one factor from each group to make them non-redundant
    if nonredundant
        for k in keys(factor_options)
            if missing ∈ factor_options[k]
                delete!(factor_options[k], missing)
            else
                delete!(factor_options[k], first(factor_options[k]))
            end
        end
    end

    # assign indexes to factors
    nextidx = 1
    factor_names = String[]
    factor_idx = Dict{Tuple{String, String}, Int}()
    for (factor, options) in factor_options
        for option in options
            factor_idx[(factor, option)] = nextidx
            push!(factor_names, string(factor, ":", option))
            nextidx += 1
        end
    end

    num_samples = length(loaded_samples.sample_names)
    num_factors = length(factor_idx)
    F = zeros(Float32, (num_samples, num_factors))

    for (i, sample_factors) in enumerate(loaded_samples.sample_factors)
        for factor in factors
            option = get(sample_factors, factor, missing)
            if haskey(factor_idx, (factor, option))
                F[i, factor_idx[(factor, option)]] = 1
            end
        end
    end

    return F, factor_names
end


function write_regression_effects(
        output_filename,
        factor_names, feature_names,
        qw_loc, qw_scale, q0, q1)

    @assert size(qw_loc) == size(qw_scale)
    num_features, num_factors = size(qw_loc)

    open(output_filename, "w") do output
        println(output, "factor,transcript_id,effect,lower_credible,upper_credible")
        for i in 1:num_features, j in 1:num_factors
            dist = Normal(qw_loc[i,j], qw_scale[i,j])
            lc = quantile(dist, q0)
            uc = quantile(dist, q1)

            @printf(
                output, "%s,%s,%f,%f,%f\n",
                factor_names[j], feature_names[i], qw_loc[i,j], lc, uc)
        end
    end
end



main()


