
import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
import Polee

include(joinpath(dirname(pathof(Polee)), "PoleeModel.jl"))
using .PoleeModel

using ArgParse
using YAML
using PyCall
using Statistics
using StatsFuns
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
    "--effect-size"
        metavar = "S"
        help = "Output the posterior probability of abs log2 fold-change greater than S"
        default = nothing
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

    if feature ∈ ["splicing"]
        error(string(parsed_args["feature"], " feature is not yet implemented."))
    end

    ts, ts_metadata = load_transcripts_from_args(parsed_args)
    n = length(ts)

    init_python_modules()
    # polee_regression_py = pyimport("polee_regression2")
    polee_regression_py = pyimport("polee_regression")
    polee_py = pyimport("polee")
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

    if feature == "gene"
        # approximate genes expression likelihood
        num_features, gene_idxs, transcript_idxs, gene_ids, gene_names =
            Polee.gene_feature_matrix(ts, ts_metadata)

        p = sortperm(gene_idxs)
        permute!(gene_idxs, p)
        permute!(transcript_idxs, p)

        feature_names = gene_ids
        feature_names_label = "gene_id"

        # idx = findfirst(isequal("gene:ENSG00000285857"), feature_names)
        # idx = findfirst(isequal("gene:ENSG00000211623"), feature_names)
        # idx = findfirst(isequal("gene:ENSG00000241756"), feature_names)
        # idx = findfirst(isequal("gene:ENSG00000248651"), feature_names)
        # idx = findfirst(isequal("gene:ENSG00000242326"), feature_names)
        # idx = findfirst(isequal("gene:ENSG00000230882"), feature_names)
        # idx = findfirst(isequal("gene:ENSG00000261058"), feature_names)
        # idx = findfirst(isequal("gene:ENSG00000183423"), feature_names)
        # idx = findfirst(isequal("gene:ENSG00000075624"), feature_names)
        idx = findfirst(isequal("gene:ENSG00000135269"), feature_names)
        @show idx

        # @show transcript_idxs[gene_idxs .== idx]

        # exit()

        # pm = Polee.posterior_mean(loaded_samples)
        # pm ./= loaded_samples.efflen_values
        # pm ./= sum(pm, dims=2)

        # qs = Float64[]
        # for i in 1:size(pm, 1)
        #     push!(qs, quantile(pm[i,:], 0.9999))
        # end

        # @show size(exp.(mean(log.(pm), dims=1)))
        # @show exp.(mean(log.(pm), dims=1))[1,1:3]
        # @show (pm ./ exp.(mean(log.(pm), dims=1)))[:,1:3]

        # @show median(pm ./ exp.(mean(log.(pm), dims=1)), dims=2)

        # @show qs
        # @show qs ./ median(qs)
        # exit()

        sess = tf.Session()

        # tf.set_random_seed(1)

        qx_gene_loc, qx_gene_scale = polee_py.approximate_feature_likelihood(
            loaded_samples.init_feed_dict,
            loaded_samples.variables,
            num_samples, num_features, n,
            gene_idxs .- 1, transcript_idxs .- 1,
            sess)

        @show qx_gene_loc[:,idx]
        @show qx_gene_scale[:,idx]

        # exit()

        # TODO: Outputing a bunch of junk to check that point estimate are 
        # at least similar.

        # pm = Polee.posterior_mean(loaded_samples)
        # pm ./= loaded_samples.efflen_values
        # pm ./= sum(pm, dims=2)
        # open("transcript-post-mean.csv", "w") do output
        #     println(output, "transcript_id,tpm")
        #     for (t, x) in zip(ts, pm[1,:])
        #         println(output, t.metadata.name, ",", 1e6 * x)
        #     end
        # end

        # open("alt-transcript-post-mean.csv", "w") do output
        #     println(output, "transcript_id,tpm")
        #     for (t, x) in zip(ts, qx_gene_loc[1,:])
        #         println(output, t.metadata.name, ",", 1e6*x)
        #     end
        # end

        # open("gene-post-mean.csv", "w") do output
        #     println(output, "gene_id,tpm")
        #     for (gene_id, x) in zip(feature_names, qx_gene_loc[1,:])
        #         println(output, gene_id, ",", 1e6*exp(x))
        #     end
        # end

        # idx = findfirst(isequal("gene:ENSG00000248537"), gene_ids)
        # idx = findfirst(isequal("gene:ENSG00000135269"), gene_ids)
        # @show idx
        # @show qx_gene_loc[:,idx]
        # @show qx_gene_scale[:,idx]
        # exit(1)

        tf.reset_default_graph()
        sess.close()

        qx_loc, qw_loc, qw_scale, qx_bias, qx_scale, =
            polee_regression_py.estimate_feature_linear_regression(
                loaded_samples.init_feed_dict, qx_gene_loc, qx_gene_scale,
                x0_log, factor_matrix, parsed_args["point-estimates"])

        # qw_loc, qw_scale, qx_bias, qx_scale, =
        #     polee_regression_py.estimate_feature_linear_regression(
        #         loaded_samples.variables, n, loaded_samples.init_feed_dict,
        #         gene_idxs, transcript_idxs,
        #         factor_matrix, parsed_args["point-estimates"])

        # TODO: What I want to output is the actual mean of qx, not qx_bias

        qx_mean = mean(qx_loc, dims=1)

        open("gene-mean-vs-sd.csv", "w") do output
            println(output, "gene_id,mean,sd")
            for i in 1:size(qx_bias, 1)
                # println(output, feature_names[i], ",", qx_bias[i], ",", qx_scale[i])
                println(output, feature_names[i], ",", qx_mean[i], ",", qx_scale[i])
            end
        end

        # @show qw_loc[idx]
        # @show qw_scale[idx]

    elseif feature == "transcript"
        qw_loc, qw_scale, qx_bias, qx_scale =
            polee_regression_py.estimate_transcript_linear_regression(
                loaded_samples.init_feed_dict, loaded_samples.variables,
                x0_log, factor_matrix, parsed_args["point-estimates"])

        open("transcript-mean-vs-sd.csv", "w") do output
            println(output, "mean,sd")
            for i in 1:size(qx_bias, 1)
                println(output, qx_bias[i], ",", qx_scale[i])
            end
        end

        feature_names = String[t.metadata.name for t in ts]
        feature_names_label = "transcript_id"
    end


    write_regression_effects(
        parsed_args["output"],
        factor_names,
        feature_names_label,
        feature_names,
        qw_loc, qw_scale,
        parsed_args["lower-credible"],
        parsed_args["upper-credible"],
        parsed_args["effect-size"])
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
        factor_names, feature_names_label, feature_names,
        qw_loc, qw_scale, q0, q1, effect_size)

    @assert size(qw_loc) == size(qw_scale)
    num_features, num_factors = size(qw_loc)

    ln2 = log(2f0)

    open(output_filename, "w") do output
        print(output, "factor,", feature_names_label, ",post_mean_effect,lower_credible,upper_credible")
        if effect_size !== nothing
            print(output, ",prob_de")
            effect_size = log(abs(effect_size))
        end
        println(output)
        for i in 1:num_features, j in 1:num_factors

            # dist = Normal(qw_loc[i,j], qw_scale[i,j])
            # lc = quantile(dist, q0)
            # uc = quantile(dist, q1)

            dist = TDist(10.0)
            # dist = Normal(0.0, 1.0)
            lc = quantile(dist, q0) * qw_scale[i,j] + qw_loc[i,j]
            uc = quantile(dist, q1) * qw_scale[i,j] + qw_loc[i,j]

            @printf(
                output, "%s,%s,%f,%f,%f",
                factor_names[j], feature_names[i],
                qw_loc[i,j]/ln2, lc/ln2, uc/ln2)
            if effect_size !== nothing
                # @printf(output, ",%e", cdf(dist, -effect_size) + (1 - cdf(dist, effect_size)))
                @printf(output, ",%f",
                    exp(logaddexp(
                        logcdf(dist, (-effect_size - qw_loc[i,j]) / qw_scale[i,j]),
                        logccdf(dist, (effect_size - qw_loc[i,j]) / qw_scale[i,j]))))
                    # exp(logaddexp(
                    #     logcdf(dist, -effect_size),
                    #     logccdf(dist, effect_size))))
            end
            println(output)
        end
    end
end



main()


