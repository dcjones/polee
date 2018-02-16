
function estimate_simple_linear_regression(input::ModelInput)
    if input.feature == :transcript
        estimate_simple_transcript_linear_regression(input)
    else
        error("Linear regression estimates for $(input.feature) not supported")
    end
end



function estimate_simple_transcript_linear_regression(input)
    num_samples, n = size(input.loaded_samples.x0_values)
    num_factors, factoridx, X = build_linear_regression_design_matrix(input)

    x0, _ = estimate_transcript_expression(input, false)

    # model specification
    # -------------------

    w_mu0 = 0.0
    w_sigma0 = 1.0
    w_bias_mu0 = log(1/n)
    w_bias_sigma0 = 2.5
    # w_bias_sigma0 = 5.0

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

    println("Estimating...")

    datadict = PyDict(Dict(x => x0))

    qw_param = tf.Variable(w_mu)
    qw_map = edmodels.PointMass(params=qw_param)

    qx_sigma_sq_param = tf.Variable(tf.fill([n], 0.0f0), name="qx_log_sigma_param")
    qx_sigma_sq_map = edmodels.PointMass(params=qx_sigma_sq_param)

    inference = ed.MAP(Dict(w => qw_map, x_sigma_sq => qx_sigma_sq_map), data=datadict)

    #optimizer = tf.train[:MomentumOptimizer](1e-7, 0.9)
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
                  lower_credible,
                  upper_credible,
                  error_sigma,
                  input.feature)
end

EXTRUDER_MODELS["simple-linear-regression"] = estimate_simple_linear_regression
