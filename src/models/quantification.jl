
function tf_print_span(var, label)
    return tf.Print(var, [tf.reduce_min(var), tf.reduce_max(var)], label)
end


function estimate_expression(input::ModelInput)
    if input.feature == :transcript
        return estimate_transcript_expression(input)
    elseif input.feature == :gene
        return estimate_gene_expression(input)
    elseif input.feature == :splicing
        return estimate_splicing_proportions(input)
    else
        error("Expression estimates for $(input.feature) not supported")
    end
end


function estimate_simple_expression(input::ModelInput)
    if input.feature == :transcript
        return estimate_transcript_expression(input, pooled_means=false)
    else
        error("Expression estimates for $(input.feature) not supported")
    end
end


function estimate_transcript_expression(
        input::ModelInput, write_results::Bool=true;
        pooled_means::Bool=true)

    num_samples, n = size(input.loaded_samples.x0_values)

    x0_log = log.(input.loaded_samples.x0_values)

    x, x_sigma_sq, x_mu_param, x_mu, likapprox =
        transcript_quantification_model(input, pooled_means=pooled_means)

    println("Estimating...")

    qx_mu_param = tf.Variable(x0_log)
    qx_softplus_sigma_param = tf.Variable(tf.fill([num_samples, n], -1.0f0))
    qx = edmodels.NormalWithSoftplusScale(loc=qx_mu_param, scale=qx_softplus_sigma_param)

    qx_mu_mu_param = tf.Variable(mean(x0_log, 1)[1,:])
    qx_mu_softplus_sigma_param = tf.Variable(tf.fill([n], -1.0f0))
    qx_mu = edmodels.NormalWithSoftplusScale(loc=qx_mu_mu_param, scale=qx_mu_softplus_sigma_param)

    qx_sigma_sq_mu_param    = tf.Variable(tf.fill([n], 0.0f0), name="qx_sigma_sq_mu_param")
    qx_sigma_sq_sigma_param = tf.Variable(tf.fill([n], 1.0f0), name="qx_sigma_sq_sigma_param")
    qx_sigma_sq = edmodels.TransformedDistribution(
        distribution=edmodels.NormalWithSoftplusScale(qx_sigma_sq_mu_param, qx_sigma_sq_sigma_param),
        bijector=tfdist.bijectors[:Exp](),
        name="LogNormalTransformedDistribution")

    inference = ed.KLqp(Dict(x => qx, x_mu => qx_mu, x_sigma_sq => qx_sigma_sq),
                        data=Dict(likapprox => Float32[]))
    optimizer = tf.train[:AdamOptimizer](0.02)
    run_inference(input, inference, 500, optimizer)

    sess = ed.get_session()

    mean_est  = sess[:run](qx_mu_param)
    sigma_est = sess[:run](tf.nn[:softplus](qx_softplus_sigma_param))

    lower_credible = similar(mean_est)
    upper_credible = similar(mean_est)
    for i in 1:size(mean_est, 1)
        for j in 1:size(mean_est, 2)
            dist = Normal(mean_est[i, j], sigma_est[i, j])

            lower_credible[i, j] = quantile(dist, input.credible_interval[1])
            upper_credible[i, j] = quantile(dist, input.credible_interval[2])
        end
    end

    # TODO: this should be a temporary measure until we decide exactly how
    # results should be reported. Probably in sqlite or something.
    if write_results
        output_filename = isnull(input.output_filename) ?
            "transcript-expression-estimates.csv" : get(input.output_filename)
        write_transcript_expression_csv(output_filename,
                                        input.ts, input.loaded_samples.sample_names,
                                        mean_est, lower_credible, upper_credible)
    end

    # open("efflen.csv", "w") do out
    #     println(out, "transcript_num,efflen")
    #     for (i, efflen) in enumerate(efflens)
    #         println(out, i, ",", efflen)
    #     end
    # end

    return mean_est, sigma_est
end


"""
Estimate transcript expression treating each sample as independent. (As
apposed to using pooled mean and variance parameters as
`estimate_transcript_expression``)
"""
function estimate_simple_transcript_expression(input::ModelInput, write_results::Bool=true)
    num_samples, n = size(input.loaded_samples.x0_values)

    x = edmodels.MultivariateNormalDiag(tf.constant(log(1/n), shape=[num_samples, n]),
                                        tf.constant(10.0, shape=[num_samples, n]))

    likapprox = RNASeqApproxLikelihood(input, x)

    x0_log = tf.log(tf.constant(input.loaded_samples.x0_values))
    qx_mu_param = tf.Variable(x0_log)
    qx_sigma_param = tf.nn[:softplus](tf.Variable(tf.fill([num_samples, n], -1.0f0)))
    qx = edmodels.MultivariateNormalDiag(qx_mu_param, qx_sigma_param)
    inference = ed.KLqp(Dict(x => qx),
                        data=Dict(likapprox => Float32[]))
    optimizer = tf.train[:AdamOptimizer](0.05)
    run_inference(input, inference, 500, optimizer)

    sess = ed.get_session()

    mean_est  = sess[:run](qx_mu_param)
    sigma_est = sess[:run](qx_sigma_param)

    lower_credible = similar(mean_est)
    upper_credible = similar(mean_est)
    for i in 1:size(mean_est, 1)
        for j in 1:size(mean_est, 2)
            dist = Normal(mean_est[i, j], sigma_est[i, j])

            lower_credible[i, j] = quantile(dist, input.credible_interval[1])
            upper_credible[i, j] = quantile(dist, input.credible_interval[2])
        end
    end

    if write_results
        # TODO: option to toggle output type
        # write_transcript_expression_csv("estimates.csv",
        #                                 input.loaded_samples.sample_names,
        #                                 mean_est, lower_credible, upper_credible)

        # TODO: figure out best way to generate credible intervals
        write_simplified_transcript_expression_csv("estimates.csv", input, mean_est)
    end

    return mean_est, sigma_est
end


function write_transcript_expression_csv(output_filename, ts, sample_names,
                                         est, lower_credible, upper_credible)
    num_samples, n = size(est)
    open(output_filename, "w") do output
        println(output, "sample_name,transcript_id,expression,lower_credible,upper_credible")
        for (i, sample_name) in enumerate(sample_names)
            for (j, t) in enumerate(ts)
                @printf(output, "%s,%s,%e,%e,%e\n", sample_name, t.metadata.name,
                        est[i, j], lower_credible[i, j], upper_credible[i, j])
            end
        end
    end
end


function write_simplified_transcript_expression_csv(output_filename,
                                                    input::ModelInput, est)
    num_samples, n = size(est)

    x = exp.(est)
    x ./= sum(x, 1)
    x .*= 1e6

    open(output_filename, "w") do output
        println(output, "sample_name,gene_id,gene_name,transcript_id,tpm")
        for (i, sample_name) in enumerate(input.loaded_samples.sample_names)
            for (j, t) in enumerate(input.ts)
                @printf(output, "%s,%s,%s,%s,%0.4f\n",
                        sample_name, t.metadata.name,
                        input.ts_metadata.gene_id[t.metadata.name],
                        input.ts_metadata.gene_name[t.metadata.name],
                        x[i, j])
            end
        end
    end
end


"""
Convert transcript expression vectors to feature (e.g. gene) expression
vectors by suming.
"""
function transcript_to_feature_expression_transform(feature_idxs, transcript_idxs, x)
    # I could do this with gather_nd more efficiently, but that would be
    # relying on the unguranteed behavior of indexes into the same cell being
    # summed. Sparse matrix multiply is safer.

    feature_matrix = tf.SparseTensor(
        indices=hcat(feature_idxs, transcript_idxs),
        values=tf.ones(length(feature_idxs)),
        dense_shape=[num_features, n])

    feature_xs = Any[]
    for x_i in tf.unstack(x)
        push!(feature_xs, tf.sparse_tensor_dense_matmul(feature_matrix, x_i))

        # TODO: should these then be normalized to sum to 1?
    end

    return tf.squeeze(tf.stack(xs), axis=-1)
end


function estimate_gene_expression(input::ModelInput)
    num_samples, n = size(input.loaded_samples.x0_values)
    num_features, gene_idxs, transcript_idxs, gene_ids, gene_names =
        gene_feature_matrix(input.ts, input.ts_metadata)
    num_aux_features = regularize_disjoint_feature_matrix!(gene_idxs, transcript_idxs, n)

    prior_vars, prior_var_approximations =
        model_disjoint_feature_prior(input, gene_idxs, transcript_idxs)

    vars, var_approximations, data =
        model_disjoint_feature_expression(input, gene_idxs, transcript_idxs,
                                          num_features + num_aux_features,
                                          prior_vars[:x_feature])

    merge!(var_approximations, prior_var_approximations)
    merge!(vars, prior_vars)

    inference = ed.KLqp(latent_vars=var_approximations, data=data)
    optimizer = tf.train[:AdamOptimizer](5e-2)
    run_inference(input, inference, 1000, optimizer)

    sess = ed.get_session()
    est  = sess[:run](tf.nn[:softmax](var_approximations[vars[:x_feature]][:mean]()))

    if input.output_format == :csv
        output_filename = isnull(input.output_filename) ? "gene-expression.csv" : get(input.output_filename)
        write_gene_expression_csv(output_filename, input.loaded_samples.sample_names,
                                  gene_ids, gene_names, num_aux_features, est)
    elseif input.output_format == :sqlite3
        error("Sqlite3 output for gene expression is not implemented.")
    end
end


function write_gene_expression_csv(output_filename, sample_names,
                                   gene_ids, gene_names,
                                   num_aux_features, est)
    n = size(est, 2) - num_aux_features
    @assert length(gene_names) == n
    open(output_filename, "w") do output
        println(output, "sample_name,gene_id,gene_name,tpm")
        for (i, sample_name) in enumerate(sample_names)
            for (j, gene_name) in enumerate(gene_names)
                @printf(output, "%s,%s,%s,%e\n", sample_name,
                        gene_ids[j], gene_name, 1e6 * est[i,j])
            end
        end
    end
end


function estimate_splicing_proportions(input::ModelInput; write_results::Bool=true)

    splice_loc_param, splice_scale_param = approximate_splicing_likelihood(input)

    # TODO: we could try to build a heirarchical model here, instead of just taking
    # quantiles of the likelihood function.

    qx_feature = edmodels.TransformedDistribution(
        distribution=edmodels.NormalWithSoftplusScale(
            loc=splice_loc_param,
            scale=splice_scale_param),
        bijector=tfdist.bijectors[:Sigmoid]())

    sess = ed.get_session()

    lower_credible = sess[:run](qx_feature[:quantile](input.credible_interval[1]))
    upper_credible = sess[:run](qx_feature[:quantile](input.credible_interval[2]))
    est = sess[:run](qx_feature[:quantile](0.5))

    if write_results
        output_filename = isnull(input.output_filename) ?
            "splicing-proportion-estimates.csv" : get(input.output_filename)

        write_splicing_proportions_csv(output_filename, input.loaded_samples.sample_names,
                                    est, lower_credible, upper_credible)
    end

    return est
end


function write_splicing_proportions_csv(output_filename, sample_names,
                                        est, lower_credible, upper_credible)
    num_features = size(est, 2)
    open(output_filename, "w") do output
        println(output, "sample_name,feature_num,proportion,lower_credible,upper_credible")
        for (i, sample_name) in enumerate(sample_names)
            for j in 1:num_features
                @printf(output, "%s,%d,%e,%e,%e\n", sample_name, j,
                        est[i, j], lower_credible[i, j], upper_credible[i, j])
            end
        end
    end
end


"""
Build basic transcript quantification model with some shrinkage towards a pooled mean.
"""
function transcript_quantification_model(input::ModelInput; pooled_means::Bool=true)
    num_samples, n = size(input.loaded_samples.x0_values)

    x_mu_mu0 = tf.constant(log(1f0/n), shape=[n])
    x_mu_sigma0 = tf.constant(4.0f0, shape=[n])
    x_mu = edmodels.Normal(loc=x_mu_mu0, scale=x_mu_sigma0)

    x_sigma_alpha0 = tf.constant(SIGMA_ALPHA0, shape=[n])
    x_sigma_beta0 = tf.constant(SIGMA_BETA0, shape=[n])
    x_sigma_sq = edmodels.InverseGamma(x_sigma_alpha0, x_sigma_beta0)
    x_sigma = tf.sqrt(x_sigma_sq)

    x_mu_param = tf.matmul(
        tf.ones([num_samples, 1]), tf.expand_dims(x_mu, 0))

    if pooled_means
        x = edmodels.Normal(loc=x_mu_param, scale=x_sigma)
    else
        x = edmodels.Normal(
            loc=tf.fill([num_samples, n], log(1f0/n)),
            scale=4.0f0)
    end

    likapprox = RNASeqApproxLikelihood(input, x)

    return x, x_sigma_sq, x_mu_param, x_mu, likapprox
end


"""
Set up prior on feature expression for simple quantification.
"""
function model_disjoint_feature_prior(input::ModelInput, feature_idxs, transcript_idxs)
    num_features = maximum(feature_idxs)

    num_samples, n = size(input.loaded_samples.x0_values)

    x_feature_mu_mu0 = tf.constant(log(1/num_features), shape=[num_features])
    x_feature_mu_sigma0 = tf.constant(4.0, shape=[num_features])
    x_feature_mu = edmodels.Normal(loc=x_feature_mu_mu0, scale=x_feature_mu_sigma0)

    x_feature_sigma_alpha0 = tf.constant(SIGMA_ALPHA0, shape=[num_features])
    x_feature_sigma_beta0 = tf.constant(SIGMA_BETA0, shape=[num_features])
    x_feature_sigma_sq = edmodels.InverseGamma(x_feature_sigma_alpha0,
                                               x_feature_sigma_beta0)
    x_feature_sigma = tf.sqrt(x_feature_sigma_sq)

    x_feature_mu_param = tf.matmul(tf.ones([num_samples, 1]),
                                   tf.expand_dims(x_feature_mu, 0))

    x_feature_sigma_param = tf.matmul(tf.ones([num_samples, 1]),
                                      tf.expand_dims(x_feature_sigma, 0))

    x_feature = edmodels.Normal(loc=x_feature_mu_param,
                                scale=x_feature_sigma_param)

    # approximations
    # --------------

    # figure out some reasonable initial values
    feature_mu_initial = zeros(Float32, (num_samples, num_features))
    for i in 1:num_samples
        for (j, k) in zip(feature_idxs, transcript_idxs)
            feature_mu_initial[i, j] += input.loaded_samples.x0_values[i, k]
        end
    end
    map!(log, feature_mu_initial, feature_mu_initial)
    feature_mu_initial_mean = reshape(mean(feature_mu_initial, 1), (num_features,))

    qx_feature_mu_mu_param = tf.Variable(feature_mu_initial_mean)
    qx_feature_mu_softplus_sigma_param = tf.Variable(tf.fill([num_features], -1.0f0))
    qx_feature_mu = edmodels.NormalWithSoftplusScale(loc=qx_feature_mu_mu_param,
                                                     scale=qx_feature_mu_softplus_sigma_param)

    qx_feature_sigma_sq_mu_param    = tf.Variable(tf.fill([num_features], 0.0f0), name="qx_sigma_sq_mu_param")
    qx_feature_sigma_sq_sigma_param = tf.Variable(tf.fill([num_features], 1.0f0), name="qx_sigma_sq_sigma_param")
    qx_feature_sigma_sq = edmodels.TransformedDistribution(
        distribution=edmodels.NormalWithSoftplusScale(loc=qx_feature_sigma_sq_mu_param,
                                                      scale=qx_feature_sigma_sq_sigma_param),
        bijector=tfdist.bijectors[:Exp](),
        name="LogNormalTransformedDistribution")

    qx_feature_mu_param = tf.Variable(feature_mu_initial)
    qx_feature_softplus_sigma_param = tf.Variable(tf.fill([num_samples, num_features], -1.0f0))
    qx_feature = edmodels.NormalWithSoftplusScale(loc=qx_feature_mu_param,
                                                  scale=qx_feature_softplus_sigma_param)

    prior_var_approximations = Dict(
        x_feature             => qx_feature,
        x_feature_mu          => qx_feature_mu,
        x_feature_sigma_sq    => qx_feature_sigma_sq,
    )

    prior_vars = Dict(
        :x_feature           => x_feature,
        :x_feature_mu        => x_feature_mu,
        :x_feature_sigma_sq  => x_feature_sigma_sq
    )

    return prior_vars, prior_var_approximations
end


function model_disjoint_feature_expression(input::ModelInput, feature_idxs,
                                           transcript_idxs, num_features,
                                           x_feature)

    num_samples, n = size(input.loaded_samples.x0_values)

    # within feature relative expression of feature constituents
    x_constituent_mu_mu0 = tf.constant(0.0f0, shape=[n])
    x_constituent_mu_sigma0 = tf.constant(10.0f0, shape=[n])
    x_constituent_mu = edmodels.Normal(loc=x_constituent_mu_mu0,
                                       scale=x_constituent_mu_sigma0)

    x_constituent_sigma_alpha0 = tf.constant(SIGMA_ALPHA0, shape=[n])
    x_constituent_sigma_beta0 = tf.constant(SIGMA_BETA0, shape=[n])
    x_constituent_sigma_sq = edmodels.InverseGamma(x_constituent_sigma_alpha0,
                                                   x_constituent_sigma_beta0)
    x_constituent_sigma = tf.sqrt(x_constituent_sigma_sq)

    x_constituent_mu_param = tf.matmul(tf.ones([num_samples, 1]),
                                       tf.expand_dims(x_constituent_mu, 0))

    x_constituent_sigma_param = tf.matmul(tf.ones([num_samples, 1]),
                                          tf.expand_dims(x_constituent_sigma, 0))

    x_constituent = edmodels.Normal(loc=x_constituent_mu_param,
                                    scale=x_constituent_sigma_param)

    # transcript expression as a deterministic function of feature expression
    # and constuent relative expression.
    p = sortperm(feature_idxs)
    permute!(feature_idxs, p)
    permute!(transcript_idxs, p)
    x_constituent_indices = Array{Int32}((n, 2))
    for (k, (i, j)) in enumerate(zip(feature_idxs, transcript_idxs))
        x_constituent_indices[k, 1] = i - 1
        x_constituent_indices[k, 2] = j - 1
    end

    xs = []
    for (x_feature_i, x_constituent_i) in zip(tf.unstack(x_feature), tf.unstack(x_constituent))
        x_constituent_matrix = tf.SparseTensor(indices=x_constituent_indices,
                                               values=x_constituent_i,
                                               dense_shape=[num_features, n])
        x_constituent_matrix_softmax = tf.sparse_softmax(x_constituent_matrix)

        x_i_exp = tf.sparse_tensor_dense_matmul(
            x_constituent_matrix_softmax,
            tf.expand_dims(tf.exp(x_feature_i), -1),
            adjoint_a=true)
        x_i_exp = tf.clip_by_value(x_i_exp, 1f-16, Inf32)
        x_i = tf.log(x_i_exp)

        push!(xs, x_i)
    end
    x = tf.squeeze(tf.stack(xs), axis=-1)
    likapprox = RNASeqApproxLikelihood(input, x)

    # Inference
    # ---------

    # figure out some reasonable initial values
    feature_mu_initial     = zeros(Float32, (num_samples, num_features))
    constituent_mu_initial = zeros(Float32, (num_samples, n))
    for i in 1:num_samples
        for (j, k) in zip(feature_idxs, transcript_idxs)
            feature_mu_initial[i, j] += input.loaded_samples.x0_values[i, k]
            constituent_mu_initial[i, k] = input.loaded_samples.x0_values[i, k]
        end

        for (j, k) in zip(feature_idxs, transcript_idxs)
            constituent_mu_initial[i, k] /= feature_mu_initial[i, j]
        end
    end

    vars = Dict(
        :x_constituent => x_constituent,
        :x_constituent_mu => x_constituent_mu,
        :x_constituent_sigma_sq => x_constituent_sigma_sq
    )

    map!(log, feature_mu_initial, feature_mu_initial)
    map!(log, constituent_mu_initial, constituent_mu_initial)
    feature_mu_initial_mean = reshape(mean(feature_mu_initial, 1), (num_features,))
    constituent_mu_initial_mean = reshape(mean(constituent_mu_initial, 1), (n,))

    # @show extrema(feature_mu_initial)
    # @show extrema(constituent_mu_initial)
    # @show extrema(feature_mu_initial_mean)
    # @show extrema(constituent_mu_initial_mean)
    # exit()

    if input.inference == :map
        qx_constituent_mu_param = tf.Variable(
            constituent_mu_initial_mean, name="qx_constituent_mu_param")
        qx_constituent_mu = edmodels.PointMass(qx_constituent_mu_param)

        qx_constituent_sigma_sq_param = tf.Variable(
            tf.fill([n], 0.0f0), name="qx_sigma_sq_mu_param")
        qx_constituent_sigma_sq = edmodels.PointMass(
            tf.nn[:softplus](qx_constituent_sigma_sq_param))

        qx_constituent_param = tf.Variable(
            constituent_mu_initial, name="qx_constituent_param")
        qx_constituent = edmodels.PointMass(qx_constituent_param)

    elseif input.inference == :variational
        qx_constituent_mu_mu_param = tf.Variable(constituent_mu_initial_mean)
        qx_constituent_mu_softplus_sigma_param = tf.Variable(tf.fill([n], -1.0f0))
        qx_constituent_mu = edmodels.NormalWithSoftplusScale(loc=qx_constituent_mu_mu_param,
                                                            scale=qx_constituent_mu_softplus_sigma_param)

        qx_constituent_sigma_sq_mu_param    = tf.Variable(tf.fill([n], 0.0f0), name="qx_sigma_sq_mu_param")
        qx_constituent_sigma_sq_sigma_param = tf.Variable(tf.fill([n], 1.0f0), name="qx_sigma_sq_sigma_param")
        qx_constituent_sigma_sq = edmodels.TransformedDistribution(
            distribution=edmodels.NormalWithSoftplusScale(loc=qx_constituent_sigma_sq_mu_param,
                                                        scale=qx_constituent_sigma_sq_sigma_param),
            bijector=tfdist.bijectors[:Exp](),
            name="LogNormalTransformedDistribution")

        qx_constituent_mu_param = tf.Variable(constituent_mu_initial)
        qx_constituent_softplus_sigma_param = tf.Variable(tf.fill([num_samples, n], -1.0f0))
        qx_constituent = edmodels.NormalWithSoftplusScale(loc=qx_constituent_mu_param,
                                                        scale=qx_constituent_softplus_sigma_param)
    else
        error("Inference method $(input.inference) not supported by the disjoint feature model.")
    end

    var_approximations = Dict(
        x_constituent          => qx_constituent,
        x_constituent_mu       => qx_constituent_mu,
        x_constituent_sigma_sq => qx_constituent_sigma_sq)

    data = Dict(likapprox => Float32[])

    return vars, var_approximations, data
end


POLEE_MODELS["expression"] = estimate_expression
POLEE_MODELS["simple-expression"] = estimate_simple_expression
