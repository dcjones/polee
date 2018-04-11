
function estimate_linear_regression(input::ModelInput)
    if input.feature == :transcript
        estimate_transcript_linear_regression(input)
    elseif input.feature == :splicing
        estimate_splicing_linear_regression(input)
    else
        error("Linear regression estimates for $(input.feature) not supported")
    end
end


function build_linear_regression_design_matrix(input::ModelInput, include_bias::Bool=true)
    num_samples, n = size(input.loaded_samples.x0_values)
    factoridx = Dict{String, Int}()

    if include_bias
        factoridx["bias"] = 1
    end

    for factors in input.loaded_samples.sample_factors
        for factor in factors
            get!(factoridx, factor, length(factoridx) + 1)
        end
    end

    num_factors = length(factoridx)
    X_ = zeros(Float32, (num_samples, num_factors))
    for i in 1:num_samples
        for factor in input.loaded_samples.sample_factors[i]
            j = factoridx[factor]
            X_[i, j] = 1
        end
    end
    if include_bias
        X_[:, factoridx["bias"]] = 1
    end
    X = tf.constant(X_)

    return num_factors, factoridx, X
end


function estimate_transcript_linear_regression(input::ModelInput)

    num_samples, n = size(input.loaded_samples.x0_values)
    num_factors, factoridx, X = build_linear_regression_design_matrix(input)

    # model specification
    # -------------------

    w_mu0 = 0.0f0
    w_sigma0 = 0.5f0
    w_bias_mu0 = log(1f0/n)
    w_bias_sigma0 = 4.0f0

    w_sigma = tf.concat(
                  [tf.constant(w_bias_sigma0, shape=[1, n]),
                   tf.constant(w_sigma0, shape=[num_factors-1, n])], 0)
    w_mu = tf.concat(
                  [tf.constant(w_bias_mu0, shape=[1, n]),
                   tf.constant(w_mu0, shape=[num_factors-1, n])], 0)
    w = edmodels.Normal(loc=w_mu, scale=w_sigma, name="W")

    x_mu = tf.matmul(X, w)

    x_sigma_alpha0 = tf.constant(SIGMA_ALPHA0, shape=[n])
    x_sigma_beta0  = tf.constant(SIGMA_BETA0, shape=[n])
    x_sigma_sq     = edmodels.InverseGamma(x_sigma_alpha0, x_sigma_beta0)
    x_sigma        = tf.sqrt(x_sigma_sq)

    # x = edmodels.StudentT(df=10.0f0, loc=x_mu, scale=x_sigma)
    x = edmodels.Normal(loc=x_mu, scale=x_sigma)

    likapprox = RNASeqApproxLikelihood(input, x)

    # inference
    # ---------

    println("Estimating...")

    x0_log = log.(input.loaded_samples.x0_values)
    qw_loc_init = vcat(
        mean(x0_log, 1),
        fill(0.0f0, (num_factors - 1, n)))

    if input.inference == :variational
        qx_loc = tf.Variable(x0_log)
        qx_softplus_scale = tf.Variable(tf.fill([num_samples, n], -2.0))
        qx = edmodels.NormalWithSoftplusScale(loc = qx_loc, scale = qx_softplus_scale)

        qw_loc = tf.Variable(qw_loc_init)
        qw_softplus_scale = tf.Variable(tf.fill([num_factors, n], -2.0f0))
        qw = edmodels.NormalWithSoftplusScale(loc=qw_loc, scale=qw_softplus_scale)

        qx_sigma_sq_mu_param    = tf.Variable(tf.fill([n], -2.0f0), name="qx_sigma_sq_mu_param")
        qx_sigma_sq_sigma_param = tf.Variable(tf.fill([n], -1.0f0), name="qx_sigma_sq_sigma_param")

        qx_sigma_sq_mu_param =  tf_print_span(qx_sigma_sq_mu_param, "qx_sigma_sq_mu_param")
        qx_sigma_sq_sigma_param =  tf_print_span(qx_sigma_sq_sigma_param, "qx_sigma_sq_sigma_param")

        qx_sigma_sq = edmodels.TransformedDistribution(
            distribution=edmodels.NormalWithSoftplusScale(qx_sigma_sq_mu_param, qx_sigma_sq_sigma_param),
            bijector=tfdist.bijectors[:Exp](),
            name="LogNormalTransformedDistribution")

        inference = ed.KLqp(Dict(x => qx, w => qw, x_sigma_sq => qx_sigma_sq),
                            data=Dict(likapprox => Float32[]))

        optimizer = tf.train[:AdamOptimizer](0.05)
        run_inference(input, inference, 500, optimizer)

        sess = ed.get_session()

        output_filename = isnull(input.output_filename) ?
            "effects.db" : get(input.output_filename)

        mean_est = sess[:run](qw_loc)
        sigma_est = sess[:run](tf.nn[:softplus](qw_softplus_scale))

        # log-normal distribution mean
        error_sigma = sqrt.(exp.(sess[:run](
            tf.add(qx_sigma_sq_mu_param,
                   tf.div(tf.square(tf.nn[:softplus](qx_sigma_sq_sigma_param)), 2.0f0)))))

        lower_credible = similar(mean_est)
        upper_credible = similar(mean_est)
        for i in 1:size(mean_est, 1)
            for j in 1:size(mean_est, 2)
                dist = Normal(mean_est[i, j], sigma_est[i, j])

                lower_credible[i, j] = quantile(dist, input.credible_interval[1])
                upper_credible[i, j] = quantile(dist, input.credible_interval[2])
            end
        end

        write_effects(output_filename, factoridx,
                    mean_est,
                    sigma_est,
                    lower_credible,
                    upper_credible,
                    error_sigma,
                    input.feature)

    elseif input.inference == :map
        qw_param = tf.Variable(qw_loc_init)
        qw_map = edmodels.PointMass(params=qw_param)

        qx_sigma_sq_param = tf.exp(tf.Variable(tf.fill([n], -1.0f0), name="qx_log_sigma_param"))
        qx_sigma_sq_map = edmodels.PointMass(params=qx_sigma_sq_param)

        inference = ed.MAP(Dict(w => qw_map, x_sigma_sq => qx_sigma_sq_map),
                        data=Dict(likapprox => Float32[]))

        optimizer = tf.train[:AdamOptimizer](0.05)
        run_inference(input, inference, 1500, optimizer)

        sess = ed.get_session()

        output_filename = isnull(input.output_filename) ?
            "effects.db" : get(input.output_filename)

        mean_est = sess[:run](qw_param)
        sigma_est = fill(0.0, size(mean_est))
        error_sigma = sqrt.(sess[:run](qx_sigma_sq_param))
        lower_credible = copy(mean_est)
        upper_credible = copy(mean_est)

        write_effects(output_filename, factoridx,
                    mean_est,
                    sigma_est,
                    lower_credible,
                    upper_credible,
                    error_sigma,
                    input.feature)
    else
        error("$(input.inference) inference not implemented for linear regression")
    end
end


function estimate_splicing_linear_regression(input::ModelInput)

    splice_loc_param, splice_scale_param = approximate_splicing_likelihood(input)

    num_samples, n = size(input.loaded_samples.x0_values)
    num_features = size(splice_loc_param, 2)
    num_factors, factoridx, z = build_linear_regression_design_matrix(input, false)

    # bias
    # ----

    w_bias_mu0 = 0.0f0
    w_bias_sigma0 = 10.0f0

    mu_bias = edmodels.Normal(loc=tf.fill([1, num_features], w_bias_mu0),
                              scale=tf.fill([1, num_features], w_bias_sigma0))

    # regression
    # ----------

    w = edmodels.Normal(loc=tf.zeros([num_factors, num_features]),
                        scale=tf.fill([num_factors, num_features], 2.0f0))

    mu_reg = tf.matmul(z, w)

    x_mu = tf.add(mu_bias, mu_reg)

    x_err_sigma_alpha0 = tf.constant(SIGMA_ALPHA0, shape=[1, num_features])
    x_err_sigma_beta0 = tf.constant(SIGMA_BETA0, shape=[1, num_features])
    x_err_sigma = edmodels.InverseGamma(x_err_sigma_alpha0, x_err_sigma_beta0, name="x_err_sigma")

    # x_err = edmodels.Normal(loc=tf.fill([num_samples, num_features], 0.0f0), scale=x_err_sigma, name="x_err")
    x_err = edmodels.StudentT(df=10.0, loc=tf.fill([num_samples, num_features], 0.0f0), scale=x_err_sigma, name="x_err")

    approxlik = polee_py.ApproximatedLikelihood(
        edmodels.NormalWithSoftplusScale(
            loc=splice_loc_param, scale=splice_scale_param),
        x_mu + x_err)

    data = Dict(approxlik => zeros(Float32, (num_samples, 0)))

    # inference
    # ---------

    qx_err_loc = tf.Variable(tf.zeros([num_samples, num_features]), name="qx_err_loc")
    qx_err_softplus_scale = tf.Variable(tf.fill([num_samples, num_features], -2.0), name="qx_err_softplus_scale")
    qx_err = edmodels.NormalWithSoftplusScale(loc=qx_err_loc, scale=qx_err_softplus_scale, name="qx_err")

    qx_err_sigma_mu_param =
        tf.Variable(tf.fill([1, num_features], -3.0f0), name="qx_sigma_mu_param")
    qx_err_sigma_sigma_param =
        tf.Variable(tf.fill([1, num_features], -1.0f0), name="qx_sigma_sigma_param")
    qx_err_sigma = edmodels.TransformedDistribution(
        distribution=edmodels.NormalWithSoftplusScale(
            qx_err_sigma_mu_param, qx_err_sigma_sigma_param),
        bijector=tfdist.bijectors[:Softplus](),
        name="qx_err_sigma_sq")

    qmu_bias_loc = tf.Variable(tf.reduce_mean(splice_loc_param, 0), name="qmu_bias_loc")
    qmu_bias_softplus_scale = tf.Variable(tf.fill([1, num_features], -1.0f0), name="qmu_bias_softplus_scale")
    qmu_bias = edmodels.NormalWithSoftplusScale(loc=qmu_bias_loc,
                                                scale=qmu_bias_softplus_scale,
                                                name="qmu_bias")

    qw_loc = tf.Variable(tf.zeros([num_factors, num_features]), name="qw_loc")
    qw_softplus_scale = tf.Variable(tf.fill([num_factors, num_features], -1.0f0), name="qw_softplus_scale")
    qw = edmodels.NormalWithSoftplusScale(loc=qw_loc, scale=qw_softplus_scale)

    var_approximations = Dict()
    var_approximations[x_err] = qx_err
    var_approximations[mu_bias] = qmu_bias
    var_approximations[x_err_sigma] = qx_err_sigma
    var_approximations[w] = qw

    inference = ed.KLqp(latent_vars=var_approximations, data=data)

    optimizer = tf.train[:AdamOptimizer](0.05)
    run_inference(input, inference, 2000, optimizer)

    sess = ed.get_session()

    output_filename = isnull(input.output_filename) ?
        "splicing-linear-regression-estimates.db" : get(input.output_filename)

    mean_est = sess[:run](qw_loc)
    sigma_est = sess[:run](tf.nn[:softplus](qw_softplus_scale))
    bias_est = sess[:run](qmu_bias_loc)
    lower_credible = similar(mean_est)
    upper_credible = similar(mean_est)
    est = similar(mean_est)

    for i in 1:size(mean_est, 1)
        for j in 1:size(mean_est, 2)
            dist = Normal(mean_est[i, j], sigma_est[i, j])
            logistic_bias_j = logistic(bias_est[j])

            est[i, j] = logistic(bias_est[j] + mean_est[i, j]) - logistic_bias_j

            lower_credible[i, j] =
                logistic(bias_est[j] + quantile(dist, input.credible_interval[1])) - logistic_bias_j
            upper_credible[i, j] =
                logistic(bias_est[j] + quantile(dist, input.credible_interval[2])) - logistic_bias_j
        end
    end

    # mean_est = sess[:run](qw_loc)
    # sigma_est = sess[:run](tf.nn[:softplus](qw_softplus_scale))
    # lower_credible = similar(mean_est)
    # upper_credible = similar(mean_est)
    # for i in 1:size(mean_est, 1)
    #     for j in 1:size(mean_est, 2)
    #         dist = Normal(mean_est[i, j], sigma_est[i, j])

    #         lower_credible[i, j] = quantile(dist, input.credible_interval[1])
    #         upper_credible[i, j] = quantile(dist, input.credible_interval[2])
    #     end
    # end

    error_sigma = sess[:run](qx_err_sigma[:quantile](0.5))

    @time write_effects(output_filename, factoridx,
                        est, sigma_est,
                        lower_credible, upper_credible,
                        error_sigma, input.feature)
end


POLEE_MODELS["linear-regression"] = estimate_linear_regression


