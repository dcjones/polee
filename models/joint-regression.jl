
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
arg_settings.prog = "polee model regression"
@add_arg_table arg_settings begin
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
    "--gene-output"
        metavar = "filename"
        help = "Output file for regression coefficients"
        default = "gene-regression-coefficients.csv"
    "--splice-output"
        metavar = "filename"
        help = "Output file for regression coefficients"
        default = "splice-regression-coefficients.csv"
    "--credible-interval"
        metavar = "C"
        help = """Size of the 0-centered credible interval to use when estimating
            minimum effect size."""
        default = 0.1
        arg_type = Float64
    "--write-variational-posterior-params"
        action = :store_true
    "--effect-size"
        metavar = "S"
        help = "Output the posterior probability of abs log2 fold-change greater than S"
        default = nothing
        arg_type = Float64
    "--factors"
        help = """
            Comma-separated list of factors to regress on. (Default: use all factors)
        """
        default = nothing
        arg_type = String
    "--nonredundant"
        help = "Avoid overparameterization by excluding one factor from each group"
        action = :store_true
    "--balanced"
        help = "Instead of factors represented as 0/1 in the design matrix, use -1/1"
        action = :store_true
    "experiment"
        metavar = "experiment.yml"
        help = "Experiment specification"
end


function main()
    parsed_args = parse_args(arg_settings)

    ts, ts_metadata = load_transcripts_from_args(parsed_args)
    n = length(ts)

    init_python_modules()

    tf_py = pyimport("tensorflow")
    tf_py.config.threading.set_inter_op_parallelism_threads(Threads.nthreads())
    tf_py.config.threading.set_intra_op_parallelism_threads(Threads.nthreads())

    polee_regression_py = pyimport("polee_regression")

    spec = YAML.load_file(parsed_args["experiment"])
    use_point_estimates = parsed_args["point-estimates"] !== nothing

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
    x0_log = log.(loaded_samples.x0_values)

    q0 = parsed_args["lower-credible"]
    q1 = parsed_args["upper-credible"]

    factors = parsed_args["factors"] === nothing ?
        nothing : split(parsed_args["factors"], ',')

    factor_matrix, factor_names = build_factor_matrix(
        loaded_samples, factors, parsed_args["nonredundant"])

    if parsed_args["balanced"]
        for idx in eachindex(factor_matrix)
            if factor_matrix[idx] == 0
                factor_matrix[idx] = -1
            end
        end
    end

    (tss_is, tss_js, tss_metadata, num_tss,
     feature_is, feature_js, feature_metadata, num_features) =
        transcript_feature_matrices(ts, ts_metadata)

    # put these in the order expected by tf.SparseTensor
    p = sortperm(collect(zip(tss_is, tss_js)))
    tss_is = tss_is[p]
    tss_js = tss_js[p]

    p = sortperm(collect(zip(feature_is, feature_js)))
    feature_is = feature_is[p]
    feature_js = feature_js[p]

    println("read ", num_tss, " transcript start sites")
    println(" and ", num_features, " splicing features")

    gene_sizes = zeros(Float32, num_tss)
    for j in tss_js
        gene_sizes[j] += 1
    end
    @assert minimum(gene_sizes) > 0

    sample_scales = estimate_sample_scales(log.(loaded_samples.x0_values), upper_quantile=0.95)

    x_gene_init, x_isoform_init = gene_initial_values(
        tss_js, tss_is,
        loaded_samples.x0_values, num_samples, num_tss, n)

    regression = polee_regression_py.RNASeqJointLinearRegression(
        loaded_samples.variables,
        tss_is, tss_js, num_tss,
        feature_is, feature_js, num_features,
        gene_sizes,
        x_gene_init, x_isoform_init,
        factor_matrix, sample_scales, use_point_estimates)

    qw_gene_loc, qw_gene_scale, w_splice_samples = regression.fit(50)

    write_gene_regression_results(
        parsed_args["gene-output"], ts, tss_metadata, w_gene_loc, w_gene_scale)

    write_splice_regression_results(
        parsed_args["splice-output"], w_splice_samples)
end


function write_gene_regression_results(
        output_filename, factor_names,
        ts, tss_metadata, qw_gene_loc, qw_gene_scale,
        min_effect_size_confidence)

    @assert size(qw_gene_loc) == size(qw_gene_scale)
    num_factors, num_features = size(qw_gene_loc)
    ln2 = log(2f0)

    tss_names = String[
        string(m.seqname, ":", m.position,
            "[", (m.strand == STRAND_POS ? "+" : "-"), "]") for m in tss_metadata]

    open(open(output_filename), "w") do output
        println(output, "factor,tss,mean_effect_size,min_effect_size")
        for i in 1:num_factors, j in 1:num_features

            min_effect_size = find_minimum_effect_size(
                qw_gene_loc[i,j], qw_gene_scale[i,j], min_effect_size_confidence)
            min_effect_size /= ln2

            println(
                output, factor_names[i], ",", tss_names[j], ",",
                qw_gene_loc[i,j] / ln2, ",", min_effect_size)
        end
    end
end


function splice_feature_description(metadata::FeatureMetadata)
    return string(
        metadata.seqname, ":",
        metadata.first "-", metadata.last,
        "[", (metadata.strand == STRAND_POS ? "+" : "-"), "]")
end


function write_splice_regression_results(
        output_filename, factor_names, feature_metadata, w_splice_samples,
        min_effect_size_confidence)

    num_factors, num_features, num_reps = size(w_splice_samples)

    splice_feature_descriptions = map(
        splice_feature_description, feature_metadata)

    splice_feature_types = [m.type for m in feature_metadata]

    open(output_filename, "w") do output
        println(
            output,
            "factor,feature,feature_type,mean_effect_size,min_effect_size")
        for i in 1:num_factors, j in 1:num_features
            min_effect_size = find_minimum_effect_size_from_samples(
                @view w_splice_samples[i, j, :], min_effect_size_confidence)
            mean_effect_size = mean(@view w_splice_samples[i, j, :])

            println(
                output,
                factor_names[i], ",",
                splice_feature_descriptions[j], ",",
                splice_feature_types[j], ",",
                mean_effect_size, ",",
                min_effect_size)
        end
    end
end


function find_minimum_effect_size(μ, σ, target_coverage)
    dist = Normal(μ, σ)

    δ_min = 0.0
    δ_max = 20.0
    coverage = 1.0
    while abs(coverage - target_coverage) > 0.001
        δ = (δ_max + δ_min) / 2
        coverage = cdf(dist, δ) - cdf(dist, -δ)

        if coverage > target_coverage
            δ_max = δ
        else
            δ_min = δ
        end
    end

    δ = (δ_max + δ_min) / 2
    return δ
end


function find_minimum_effect_size_from_samples(xs, target_coverage)
    xs = sort(abs.(xs))
    return xs[clamp(round(Int, target_coverage * length(xs)), 1, length(xs))]
end


main()

