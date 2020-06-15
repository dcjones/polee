
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
import SQLite


const arg_settings = ArgParseSettings()
arg_settings.prog = "polee model regression"
@add_arg_table arg_settings begin
    "--feature"
        metavar = "F"
        action = :store_arg
        default = "transcript"
        help = "One of transcript, gene, gene-isoform, splice-feature"
    "--point-estimates"
        help = """
            Use point estimates read from a file specified in the experiment
            instead of approximated likelihood."""
        default = nothing
        arg_type = String
    "--kallisto-bootstrap"
        help = """
            Use kallisto bootstrap samples. The sample specifications should
            have a `kallisto` key pointing to the h5 file.
            """
        action = :store_true
    "--kallisto"
        help = """
            Use kallisto maximum likelihood estimates. The sample
            specifications should have a `kallisto` key pointing to the h5
            file.
            """
        action = :store_true
    "--gene-pattern"
        metavar = "regex"
        help = """
        A regular expression to extract gene ids from from transcript ids. This
        is useful when transcriptome alignments are used with no external
        annotation, but gene names are encoded in the FASTA sequence names.
        """
        default = nothing
        arg_type = String
    "--gene-annotations"
        metavar = "filename"
        help = """
            YAML file assigning transcripts ids to genes. Useful for doing gene
            regression with transcriptome alignments.
        """
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
    "--isoform-output"
        metavar = "filename"
        help = """
            Output file for isoform regression results when 'gene-isoform'
            regression is used. """
        default = "regression-isoform-coefficients.csv"
    "--extra-params-output"
        metavar = "filename"
        help = """
            Output some additional parameter values to the given file. Mainly
            useful for debugging.
            """
        default = nothing
    "--output-expression"
        metavar = "filename"
        help = "Output expression estimates to the given file."
        default = nothing
    "--lower-credible"
        metavar = "L"
        default = 0.025
        arg_type = Float64
    "--upper-credible"
        metavar = "U"
        default = 0.975
        arg_type = Float64
    "--min-effect-size-coverage"
        metavar = "C"
        default = 0.1
        arg_type = Float64
    "--write-variational-posterior-params"
        action = :store_true
    "--effect-size"
        metavar = "S"
        help = "Output the posterior probability of abs fold-change greater than S"
        default = nothing
        arg_type = Float64
    "--factors"
        help = """
            Comma-separated list of factors to regress on. (Default: use all factors)
        """
        default = nothing
        arg_type = String
    "--no-distortion"
        help = """
            Disable 'distortion' model in regression. Enabled by default,
            this 'distortion' model tries to remove any systemic technical
            effects, but can potentially remove true effects.
        """
        action = :store_true
    "--scale-penalty"
        help = """
            Expression vectors with sum strays too far from 1.0 are
            penalized according to a normal distribution with the given std. dev.
        """
        default = 1e-3
        arg_type = Float64
    "--nonredundant"
        help = "Avoid overparameterization by excluding one factor from each group"
        action = :store_true
    "--redundant-factor"
        help = "When --nonredundant is specified, exclude this factor"
        metavar = "factor"
        default = ""
        arg_type = String
    "--balanced"
        help = "Instead of factors represented as 0/1 in the design matrix, use -1/1"
        action = :store_true
    "experiment"
        metavar = "experiment.yml"
        help = "Experiment specification"
        required = true
end


function main()
    parsed_args = parse_args(arg_settings)

    feature = parsed_args["feature"]

    if feature ∉ ["transcript", "gene", "gene-isoform", "splice-feature"]
        error(string(parsed_args["feature"], " is not a supported feature."))
    end

    if parsed_args["gene-pattern"] !== nothing && parsed_args["gene-annotations"] !== nothing
        error("At most one of --gene-pattern and --gene-annotations can be given.")
    end


    ts, ts_metadata = load_transcripts_from_args(
        parsed_args,
        gene_annotations=parsed_args["gene-annotations"],
        gene_pattern=parsed_args["gene-pattern"])
    n = length(ts)

    init_python_modules()

    tf_py = pyimport("tensorflow")
    tf_py.config.threading.set_inter_op_parallelism_threads(Threads.nthreads())
    tf_py.config.threading.set_intra_op_parallelism_threads(Threads.nthreads())

    polee_py = pyimport("polee")
    polee_regression_py = pyimport("polee_regression")

    spec = YAML.load_file(parsed_args["experiment"])
    use_point_estimates = parsed_args["point-estimates"] !== nothing || parsed_args["kallisto"]
    pseudocount = parsed_args["pseudocount"] === nothing ? 0.0 : parsed_args["pseudocount"]

    use_kallisto = parsed_args["kallisto"] || parsed_args["kallisto-bootstrap"]

    if parsed_args["kallisto"] && parsed_args["kallisto-bootstrap"]
        error("Only one of '--kallisto' and '--kallisto-bootstrap' can be used.")
    end

    if use_kallisto && parsed_args["point-estimates"] !== nothing
        error("'--use-point-estimates' in not compatible with '--kallisto' or '--kallisto-bootstrap'")
    end

    if parsed_args["kallisto"]
        loaded_samples = load_kallisto_estimates_from_specification(
            spec, ts, ts_metadata,
            pseudocount === nothing ? 0.0 : pseudocount, false)

    elseif parsed_args["kallisto-bootstrap"]
        loaded_samples = load_kallisto_estimates_from_specification(
            spec, ts, ts_metadata,
            pseudocount === nothing ? 0.0 : pseudocount, true)

    elseif parsed_args["point-estimates"] !== nothing
        loaded_samples = load_point_estimates_from_specification(
            spec, ts, ts_metadata, parsed_args["point-estimates"])

        if pseudocount !== nothing
            loaded_samples.x0_values .+= pseudocount / 1f6
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
        loaded_samples, factors,
        parsed_args["nonredundant"] ?
        parsed_args["redundant-factor"] : nothing)

    if parsed_args["balanced"]
        for idx in eachindex(factor_matrix)
            if factor_matrix[idx] == 0
                factor_matrix[idx] = -1
            end
        end
    end

    if feature == "gene"
        if parsed_args["kallisto-bootstrap"]
            error("$(feature) regression with --kallisto-bootstrap not yet implemented")
        end

        num_features, gene_idxs, transcript_idxs, gene_ids, gene_names =
            Polee.gene_feature_matrix(ts, ts_metadata)

        p = sortperm(transcript_idxs)
        # permute!(gene_idxs, p)
        permute!(transcript_idxs, p)

        feature_names = gene_ids
        feature_names_label = "gene_id"

        gene_sizes = zeros(Float32, num_features)
        for i in gene_idxs
            gene_sizes[i] += 1
        end

        x_gene_init, x_isoform_init = gene_initial_values(
            gene_idxs, transcript_idxs,
            loaded_samples.x0_values, num_samples, num_features, n)

        sample_scales = estimate_sample_scales(log.(loaded_samples.x0_values), upper_quantile=0.95)

        regression = polee_regression_py.RNASeqGeneLinearRegression(
                loaded_samples.variables,
                gene_idxs, transcript_idxs, x_gene_init, x_isoform_init,
                gene_sizes, factor_matrix, sample_scales,
                !parsed_args["no-distortion"], parsed_args["scale-penalty"],
                use_point_estimates)

        qx_loc, qw_loc, qw_scale, qx_bias, qx_scale, = regression.fit(10000)

        # qx_mean = mean(qx_loc, dims=1)

        # dump stuff for debugging
        # open("gene-mean-vs-sd.csv", "w") do output
        #     println(output, "gene_id,mean,sd")
        #     for i in 1:size(qx_bias, 1)
        #         # println(output, feature_names[i], ",", qx_bias[i], ",", qx_scale[i])
        #         println(output, feature_names[i], ",", qx_mean[i], ",", qx_scale[i])
        #     end
        # end

    elseif feature == "transcript"
        sample_scales = estimate_sample_scales(x0_log)

        if loaded_samples.log_x0_std !== nothing
            regression = polee_regression_py.RNASeqNormalTranscriptLinearRegression(
                loaded_samples.variables,
                x0_log, loaded_samples.log_x0_std,
                factor_matrix, sample_scales,
                !parsed_args["no-distortion"],
                parsed_args["scale-penalty"])
        else
            regression = polee_regression_py.RNASeqTranscriptLinearRegression(
                loaded_samples.variables,
                x0_log, factor_matrix, sample_scales,
                !parsed_args["no-distortion"],
                parsed_args["scale-penalty"],
                use_point_estimates)
        end

        qx_loc, qw_loc, qw_scale, qx_bias, qx_scale, = regression.fit(6000)

        feature_names = Array{String}(undef, length(ts))
        for t in ts
            feature_names[t.metadata.id] = t.metadata.name
        end
        feature_names_label = "transcript_id"

        # dump stuff for debugging
        #open("transcript-mean-vs-sd.csv", "w") do output
            #println(output, "mean,sd")
            #for i in 1:size(qx_bias, 1)
                #println(output, qx_bias[i], ",", qx_scale[i])
            #end
        #end

        # open("transcript-expression.csv", "w") do output
        #     println(output, "transcript_id,sample,expression")
        #     for j in 1:size(qx_loc, 2), i in 1:size(qx_loc, 1)
        #         println(output, feature_names[j], ",", i, ",", qx_loc[i,j])
        #     end
        # end

        # open("transcript-expression-naive.csv", "w") do output
        #     println(output, "transcript_id,sample,expression")
        #     for j in 1:size(x0_log, 2), i in 1:size(x0_log, 1)
        #         println(output, feature_names[j], ",", i, ",", x0_log[i,j])
        #     end
        # end
    elseif feature == "gene-isoform"
        if parsed_args["kallisto-bootstrap"]
            error("$(feature) regression with --kallisto-bootstrap not yet implemented")
        end

        num_features, gene_idxs, transcript_idxs, gene_ids, gene_names =
            Polee.gene_feature_matrix(ts, ts_metadata)

        p = sortperm(transcript_idxs)
        # permute!(gene_idxs, p)
        permute!(transcript_idxs, p)

        @assert length(transcript_idxs) == n
        @assert transcript_idxs == collect(1:n)

        gene_sizes = zeros(Float32, num_features)
        for i in gene_idxs
            gene_sizes[i] += 1
        end

        x_gene_init, x_isoform_init = gene_initial_values(
            gene_idxs, transcript_idxs,
            loaded_samples.x0_values, num_samples, num_features, n)

        # @show x_isoform_init[1:200]
        # exit()

        sample_scales = estimate_sample_scales(log.(loaded_samples.x0_values), upper_quantile=0.95)

        regression = polee_regression_py.RNASeqGeneIsoformLinearRegression(
                loaded_samples.variables,
                gene_idxs, transcript_idxs, x_gene_init, x_isoform_init,
                gene_sizes, factor_matrix, sample_scales,
                !parsed_args["no-distortion"], parsed_args["scale-penalty"],
                use_point_estimates)

        qw_gene_loc, qw_gene_scale, qw_isoform_loc, qw_isoform_scale,
            qx_isoform_bias_loc, qx_isoform_bias_scale,
            qx_gene_bias, qx_gene_scale = regression.fit(10000)

        min_effect_sizes, mean_effect_sizes = estimate_isoform_effect_sizes(
            gene_idxs, transcript_idxs, qw_isoform_loc, qw_isoform_scale,
            qx_isoform_bias_loc, qx_isoform_bias_scale)

        transcript_names = Array{String}(undef, length(ts))
        for t in ts
            transcript_names[t.metadata.id] = t.metadata.name
        end

        write_isoform_regression_effects(
            parsed_args["isoform-output"],
            gene_idxs, transcript_idxs,
            factor_names,
            gene_ids, gene_names, transcript_names,
            min_effect_sizes, mean_effect_sizes,
            qw_isoform_loc, qx_isoform_bias_loc)

        if parsed_args["extra-params-output"] !== nothing
            regression.write_other_params(parsed_args["extra-params-output"])
        end

        # TODO: debugging by output point estimates.
        # open("x-point-estimates.csv", "w") do output
        #     println(output, "sample,transcript,proportion")
        #     for i in 1:num_samples, j in 1:n
        #         println(
        #             output,
        #             i, ",",
        #             transcript_names[j], ",",
        #             exp.(x_isoform_init[i, j]))
        #     end
        # end

        # set of variables for output
        # joining ids and names here as hack to output both names
        feature_names = [string(gene_id, ",", gene_name)
            for (gene_id, gene_name) in zip(gene_ids, gene_names)]
        feature_names_label = "gene_id,gene_name"
        qw_loc = qw_gene_loc
        qw_scale = qw_gene_scale
        qx_bias = qx_gene_bias
        qx_scale = qx_gene_scale

    elseif feature == "splice-feature"
        if parsed_args["kallisto-bootstrap"]
            error("$(feature) regression with --kallisto-bootstrap not yet implemented")
        end

        # temporary database to store feature metadata
        # TODO: add an option to write this database to a file.
        gene_db = SQLite.DB()

        num_features, feature_idxs, feature_transcript_idxs,
           antifeature_idxs, antifeature_transcript_idxs =
            splicing_features(ts, ts_metadata, gene_db, alt_ends=true)

        # compute initial values
        x_features_init = zeros(Float32, (num_samples, num_features))
        for k in 1:num_samples
            for (i, j) in zip(feature_idxs, feature_transcript_idxs)
                x_features_init[k, i] += loaded_samples.x0_values[k, j]
            end
        end

        x_antifeatures_init = zeros(Float32, (num_samples, num_features))
        for k in 1:num_samples
            for (i, j) in zip(antifeature_idxs, antifeature_transcript_idxs)
                x_antifeatures_init[k, i] += loaded_samples.x0_values[k, j]
            end
        end

        x_init = log.(x_features_init) .- log.(x_antifeatures_init)
        @assert all(isfinite.(x_init))

        println("approximating splice feature likelihood")
        qx_feature_loc, qx_feature_scale = approximate_splicing_likelihood(
            loaded_samples, num_features, feature_idxs, feature_transcript_idxs,
            antifeature_idxs, antifeature_transcript_idxs)
        println("done")

        if parsed_args["no-distortion"]
            @warn "'--no-distortion' is not applicable to splice-feature regression."
        end

        regression = polee_regression_py.RNASeqSpliceFeatureLinearRegression(
            factor_matrix, x_init, qx_feature_loc, qx_feature_scale)

        qw_loc, qw_scale, qx_bias_loc = regression.fit(10000)

        # output results
        write_splice_feature_regression_effects(
            parsed_args["output"],
            factor_names,
            qw_loc, qw_scale,
            gene_db,
            parsed_args["min-effect-size-coverage"],
            [t.metadata.name for t in ts],
            ts_metadata, x_init)

        exit()
    end

    if parsed_args["output-expression"] !== nothing
        x_tpm = exp.(qx_loc)
        x_tpm ./= sum(x_tpm, dims=2)
        x_tpm .*= 1e6
        open(parsed_args["output-expression"], "w") do output
            println(output, feature_names_label, ",sample,tpm")
            for j in 1:size(x_tpm, 2), i in 1:size(x_tpm, 1)
                println(
                    output,
                    feature_names[j], ",",
                    loaded_samples.sample_names[i], ",",
                    x_tpm[i,j])
            end
        end
    end

    write_regression_effects(
        parsed_args["output"],
        factor_names,
        feature_names_label,
        feature_names,
        qx_bias, qx_scale,
        qw_loc, qw_scale,
        parsed_args["lower-credible"],
        parsed_args["upper-credible"],
        parsed_args["effect-size"],
        parsed_args["min-effect-size-coverage"],
        parsed_args["write-variational-posterior-params"])
end


function find_minimum_effect_size(μ, σ, target_coverage)
    dist = Normal(μ, σ)

    δ_min = 0.0
    δ_max = 20.0
    coverage = 1.0
    while abs(coverage - target_coverage) / target_coverage > 0.001
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


function write_regression_effects(
        output_filename,
        factor_names, feature_names_label, feature_names,
        qx_bias, qx_scale, qw_loc, qw_scale, q0, q1, effect_size,
        mes_target_coverage, write_variational_posterior_params)

    @assert size(qw_loc) == size(qw_scale)
    num_factors, num_features = size(qw_loc)

    ln2 = log(2f0)

    open(output_filename, "w") do output
        print(
            output, "factor,", feature_names_label,
            ",min_effect_size,post_mean_effect,lower_credible,upper_credible")
        if effect_size !== nothing
            print(output, ",prob_de,prob_down_de,prob_up_de")
            effect_size = log(abs(effect_size))
        end
        if write_variational_posterior_params
            print(output, ",qx_bias_loc,qx_scale,qw_loc,qw_scale")
        end
        println(output)
        for i in 1:num_factors, j in 1:num_features

            # Using t-distribution for what is actually Normal just to avoid
            # 1.0 probabilities.
            dist = TDist(10.0)
            # dist = Normal()
            lc = quantile(dist, q0) * qw_scale[i,j] + qw_loc[i,j]
            uc = quantile(dist, q1) * qw_scale[i,j] + qw_loc[i,j]

            min_effect_size = find_minimum_effect_size(
                qw_loc[i,j], qw_scale[i,j], mes_target_coverage)

            @printf(
                output, "%s,%s,%f,%f,%f,%f",
                factor_names[i], feature_names[j],
                min_effect_size/ln2,
                qw_loc[i,j]/ln2, lc/ln2, uc/ln2)
            if effect_size !== nothing
                prob_down = cdf(dist, (-effect_size - qw_loc[i,j]) / qw_scale[i,j])
                prob_up = ccdf(dist, (effect_size - qw_loc[i,j]) / qw_scale[i,j])

                # max(prob_up, prob_down) sometimes does better than the more
                # standard prob_up + prob_down. It's particularly useful because it
                # let's us specify a minimum effect size of 0.
                @printf(output, ",%f,%f,%f", max(prob_down, prob_up), prob_down, prob_up)
                # @printf(output, ",%f,%f,%f", prob_down + prob_up, prob_down, prob_up)
            end
            if write_variational_posterior_params
                @printf(
                    output, ",%f,%f,%f,%f",
                    qx_bias[j], qx_scale[j],
                    qw_loc[i,j], qw_scale[i,j])
            end
            println(output)
        end
    end
end


function write_isoform_regression_effects(
        output_filename,
        gene_idxs, transcript_idxs,
        factor_names, gene_ids, gene_names, transcript_names,
        min_effect_sizes, mean_effect_sizes,
        qw_isoform_loc, qx_isoform_bias_loc)

    transcript_gene_idx = Dict{Int, Int}()
    for (gene_idx, transcript_idx) in zip(gene_idxs, transcript_idxs)
        transcript_gene_idx[transcript_idx] = gene_idx
    end

    num_factors, n = size(min_effect_sizes)
    ln2 = log(2f0)

    # TODO: should we also output prob_de wrt to some effect size?

    open(output_filename, "w") do output
        println(output, "factor,gene_id,gene_name,transcript_id,mean_effect_size,min_effect_size,w_mean,x_bias")
        for i in 1:num_factors, j in 1:n
            println(
                output,
                factor_names[i], ",",
                gene_ids[transcript_gene_idx[j]], ",",
                gene_names[transcript_gene_idx[j]], ",",
                transcript_names[j], ",",
                mean_effect_sizes[i, j]/ln2, ",",
                min_effect_sizes[i, j]/ln2, ",",
                qw_isoform_loc[i,j], ",",
                qx_isoform_bias_loc[j])
        end
    end
end

function estimate_isoform_effect_sizes(
        gene_idxs, transcript_idxs,
        qw_loc, qw_scale, qx_bias_loc, qx_bias_scale;
        niter=100, target_coverage=0.1)

    gene_transcript_idxs = Dict{Int, Vector{Int}}()
    for (gene_idx, transcript_idx) in zip(gene_idxs, transcript_idxs)
        push!(get!(() -> Int[], gene_transcript_idxs, gene_idx), transcript_idx)
    end

    # index mapping transcript index to an array of every transcript in the
    # same gene
    index = Dict{Int, Vector{Int}}()
    for (gene_idx, transcript_idx) in zip(gene_idxs, transcript_idxs)
        index[transcript_idx] = gene_transcript_idxs[gene_idx]
    end

    num_factors, n = size(qw_loc)

    x = zeros(Float64, n)
    x_proportion = zeros(Float64, n)
    w = zeros(Float64, (num_factors, n))

    effect_size_samples = zeros(Float32, (num_factors, n, niter))

    for iter in 1:niter
        # draw sample from x_bias posterior
        for i in 1:n
            x[i] = randn() * qx_bias_scale[i] + qx_bias_loc[i]
        end

        # softmax to gene-relative isoform proportions
        for (gene_idx, transcript_idxs) in gene_transcript_idxs
            denom = 0.0
            for i in transcript_idxs
                denom += exp(x[i])
            end

            for i in transcript_idxs
                x_proportion[i] = exp(x[i]) / denom
            end
        end

        # draw sample from w posterior
        for i in 1:num_factors, j in 1:n
            w[i, j] = randn() * qw_scale[i,j] + qw_loc[i,j]
        end

        # compute effect size for each coefficient
        for i in 1:num_factors, j in 1:n
            numer = exp(x[j] + w[i,j])
            denom = numer
            for k in index[j]
                if k != j
                    denom += exp(x[k])
                end
            end
            x_alt_proportion = numer/denom

            # need to compare to baseline proportion now
            effect_size_samples[i, j, iter] =
                log(x_alt_proportion) .- log(x_proportion[j])
        end
    end

    min_effect_sizes = Array{Float32}(undef, (num_factors, n))
    for i in 1:num_factors, j in 1:n
        min_effect_sizes[i,j] = find_minimum_effect_size_from_samples(
            (@view effect_size_samples[i, j, :]), target_coverage)
    end

    mean_effect_sizes = mean(effect_size_samples, dims=3)

    return min_effect_sizes, mean_effect_sizes
end


function find_minimum_effect_size_from_samples(xs, target_coverage)
    xs = sort(abs.(xs))
    return xs[clamp(round(Int, target_coverage * length(xs)), 1, length(xs))]
end


function write_splice_feature_regression_effects(
        output_filename, factor_names, qw_loc, qw_scale, gene_db,
        min_effect_size_coverage, transcript_ids, ts_metadata, x_init)

    num_factors, num_features = size(qw_loc)
    ln2 = log(2f0)

    seen_indexes = fill(false, num_features)
    feature_types = Vector{String}(undef, num_features)
    feature_loci = Vector{String}(undef, num_features)

    query = SQLite.Query(
        gene_db,
        """
        select feature_num, type, seqname, included_first,
            included_last, excluded_first, excluded_last
        from splicing_features
        """)
    for row in query
        row.feature_num
        seen_indexes[row.feature_num] = true
        feature_types[row.feature_num] = row.type
        feature_loci[row.feature_num] =
            string(row.seqname, ":",
                min(row.included_first, row.excluded_first), "-",
                max(row.included_last, row.excluded_last))
    end
    @assert all(seen_indexes)

    feature_gene_id_sets = [Set{String}() for _ in 1:num_features]

    fill!(seen_indexes, false)
    query = SQLite.Query(
        gene_db,
        """
        select feature_num, transcript_num
        from splicing_feature_including_transcripts
        """)
    for row in query
        push!(
            feature_gene_id_sets[row.feature_num],
            ts_metadata.gene_id[transcript_ids[row.transcript_num]])
        seen_indexes[row.feature_num] = true
    end
    @assert all(seen_indexes)

    fill!(seen_indexes, false)
    query = SQLite.Query(
        gene_db,
        """
        select feature_num, transcript_num
        from splicing_feature_excluding_transcripts
        """)
    for row in query
        push!(
            feature_gene_id_sets[row.feature_num],
            ts_metadata.gene_id[transcript_ids[row.transcript_num]])
        seen_indexes[row.feature_num] = true
    end
    @assert all(seen_indexes)

    feature_gene_ids = [join(ids, ';') for ids in feature_gene_id_sets]

    open(output_filename, "w") do output
        println(output, "factor,type,gene_ids,locus,mean_effect_size,min_effect_size")
        for i in 1:num_factors, j in 1:num_features
            min_effect_size = find_minimum_effect_size(
                qw_loc[i,j], qw_scale[i,j], min_effect_size_coverage)

            println(
                output,
                factor_names[i], ",",
                feature_types[j], ",",
                feature_gene_ids[j], ",",
                feature_loci[j], ",",
                qw_loc[i,j] / ln2, ",",
                min_effect_size / ln2)
        end
    end

    # TODO: debug output
    num_samples = size(x_init, 1)
    open(string(output_filename, ".post_mean.csv"), "w") do output
        println(output, "sample,type,gene_ids,locus,post_mean")
        for i in 1:num_samples, j in 1:num_features
            println(
                output,
                i, ",",
                feature_types[j], ",",
                feature_gene_ids[j], ",",
                feature_loci[j], ",",
                x_init[i,j])
        end
    end
end

main()
