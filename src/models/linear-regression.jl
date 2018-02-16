
function estimate_linear_regression(input::ModelInput)
    if input.feature == :transcript
        estimate_transcript_linear_regression(input)
    elseif input.feature == :splicing
        estimate_splicing_linear_regression(input)
    else
        error("Linear regression estimates for $(input.feature) not supported")
    end
end


function build_linear_regression_design_matrix(input::ModelInput)
    num_samples, n = size(input.loaded_samples.x0_values)
    factoridx = Dict{String, Int}()
    factoridx["bias"] = 1
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
    X_[:, factoridx["bias"]] = 1
    X = tf.constant(X_)

    return num_factors, factoridx, X
end


function estimate_transcript_linear_regression(input::ModelInput)

    num_samples, n = size(input.loaded_samples.x0_values)
    num_factors, factoridx, X = build_linear_regression_design_matrix(input)

    # model specification
    # -------------------

    w_mu0 = 0.0
    w_sigma0 = 1.0
    w_bias_mu0 = log(1/n)
    w_bias_sigma0 = 2.5

    w_sigma = tf.concat(
                  [tf.constant(w_bias_sigma0, shape=[1, n]),
                   tf.constant(w_sigma0, shape=[num_factors-1, n])], 0)
    w_mu = tf.concat(
                  [tf.constant(w_bias_mu0, shape=[1, n]),
                   tf.constant(w_mu0, shape=[num_factors-1, n])], 0)

    w = edmodels.MultivariateNormalDiag(name="W", w_mu, w_sigma)
    x_mu = tf.matmul(X, w)

    # x_log_sigma_mu0 = tf.constant(-1.0, shape=[n])
    # x_log_sigma_sigma0 = tf.constant(1.0, shape=[n])
    # x_log_sigma = edmodels.MultivariateNormalDiag(x_log_sigma_mu0,
    #                                               x_log_sigma_sigma0)
    # x_sigma = tf.exp(x_log_sigma)

    x_sigma_alpha0 = tf.constant(SIGMA_ALPHA0, shape=[n])
    x_sigma_beta0 = tf.constant(SIGMA_BETA0, shape=[n])
    x_sigma_sq = edmodels.InverseGamma(x_sigma_alpha0, x_sigma_beta0)
    x_sigma = tf.sqrt(x_sigma_sq)

    x = edmodels.StudentT(df=10.0, loc=x_mu, scale=x_sigma)

    likapprox = RNASeqApproxLikelihood(input, x)

    # inference
    # ---------

    println("Estimating...")

    x0_log = log.(input.loaded_samples.x0_values)
    qw_loc_init = vcat(
        mean(x0_log, 1),
        fill(0.0f0, (num_factors - 1, n)))

    qw_loc = tf.Variable(qw_loc_init)


    qw_scale = tf.nn[:softplus](tf.Variable(tf.fill([num_factors, n], -1.0)))
    qw = edmodels.MultivariateNormalDiag(qw_loc, qw_scale)

    # qx_log_sigma_mu_param = tf.Variable(tf.fill([n], 0.0f0), name="qx_log_sigma_mu_param")
    # qx_log_sigma_sigma_param = tf.nn[:softplus](tf.Variable(tf.fill([n], -1.0f0), name="qx_log_sigma_sigma_param"))
    # qx_log_sigma = edmodels.MultivariateNormalDiag(qx_log_sigma_mu_param,
    #                                                qx_log_sigma_sigma_param)

    qx_sigma_sq_mu_param    = tf.Variable(tf.fill([n], 0.0f0), name="qx_sigma_sq_mu_param")
    qx_sigma_sq_sigma_param = tf.Variable(tf.fill([n], 1.0f0), name="qx_sigma_sq_sigma_param")
    qx_sigma_sq = edmodels.TransformedDistribution(
        distribution=edmodels.NormalWithSoftplusScale(qx_sigma_sq_mu_param, qx_sigma_sq_sigma_param),
        bijector=tfdist.bijectors[:Exp](),
        name="LogNormalTransformedDistribution")

    qx_mu_param = tf.Variable(x0_log, name="qx_mu_param")
    qx_sigma_param = tf.nn[:softplus](tf.Variable(tf.fill([num_samples, n], -1.0f0), name="qx_sigma_param"))
    qx = edmodels.MultivariateNormalDiag(qx_mu_param, qx_sigma_param)

    inference = ed.KLqp(Dict(w => qw, x => qx, x_sigma_sq => qx_sigma_sq),
                        data=Dict(likapprox => Float32[]))

    optimizer = tf.train[:AdamOptimizer](0.05)
    run_inference(input, inference, 1500, optimizer)

    # run_inference(input, inference, 20, optimizer)

    sess = ed.get_session()

    output_filename = isnull(input.output_filename) ?
        "effects.db" : get(input.output_filename)

    mean_est = sess[:run](qw_loc)
    sigma_est = sess[:run](qw_scale)
    error_sigma = sqrt.(exp.(sess[:run](qx_sigma_sq_mu_param)))
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
                  lower_credible,
                  upper_credible,
                  error_sigma,
                  input.feature)
end


function estimate_splicing_linear_regression(input::ModelInput)
    num_samples, n = size(input.loaded_samples.x0_values)
    num_factors, factoridx, X = build_linear_regression_design_matrix(input)

    (num_features,
     feature_idxs, feature_transcript_idxs,
     antifeature_idxs, antifeature_transcript_idxs) = splicing_features(input)

    # Model relative feature expression with linear regression

    w_mu0 = 0.0
    w_sigma0 = 1.0
    w_bias_mu0 = log(1/num_features)
    w_bias_sigma0 = 5.0

    w_sigma = tf.concat(
                  [tf.constant(w_bias_sigma0, shape=[1, num_features]),
                   tf.constant(w_sigma0, shape=[num_factors-1, num_features])], 0)
    w_mu = tf.concat(
                  [tf.constant(w_bias_mu0, shape=[1, num_features]),
                   tf.constant(w_mu0, shape=[num_factors-1, num_features])], 0)

    w = edmodels.MultivariateNormalDiag(name="W", w_mu, w_sigma)

    x_feature = tf.matmul(X, w)


    vars, var_approximations, data =
        model_nondisjoint_feature_expression(input, num_features,
                                            feature_idxs, feature_transcript_idxs,
                                            antifeature_idxs, antifeature_transcript_idxs,
                                            x_feature)

    qw_loc = tf.Variable(tf.multiply(0.001, tf.random_normal([num_factors, num_features])))
    qw_scale = tf.nn[:softplus](tf.Variable(tf.zeros([num_factors, num_features])))
    qw = edmodels.Normal(loc=qw_loc, scale=qw_scale)

    vars[:w] = w
    var_approximations[w] = qw

    inference = ed.KLqp(latent_vars=var_approximations, data=data)

    optimizer = tf.train[:AdamOptimizer](5e-2)
    run_inference(input, inference, 2000, optimizer)

    sess = ed.get_session()

    output_filename = isnull(input.output_filename) ?
        "effects.db" : get(input.output_filename)

    # TODO: need a way to output the features and such
    @time write_effects(output_filename, factoridx,
                        sess[:run](qw_loc),
                        sess[:run](qw_scale),
                        input.feature)
end


EXTRUDER_MODELS["linear-regression"] = estimate_linear_regression


