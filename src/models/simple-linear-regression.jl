
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

    # using pooled estimates
    # x0, _ = estimate_transcript_expression(input, false)

    # using simple estimates
    x0, _ = estimate_simple_transcript_expression(input, false)

    # open("delta.csv", "w") do output
    #     for x in x0[1,:] .- x0[2,:]
    #         println(output, x)
    #     end
    # end

    # upper quantile normalization
    # ----------------------------

    qs = Array{Float32}(num_samples)
    x0 = exp.(x0)
    for i in 1:num_samples
        qs[i] = quantile(x0[i,:], 0.9)
    end
    qs ./= mean(qs)

    for i in 1:num_samples
        for j in 1:n
            x0[i,j] /= qs[i]
        end
    end
    x0 = log.(x0)

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
    w = edmodels.Normal(loc=w_mu, scale=w_sigma, name="W")

    x_mu = tf.matmul(X, w)

    x_sigma_alpha0 = tf.constant(SIGMA_ALPHA0, shape=[n])
    x_sigma_beta0  = tf.constant(SIGMA_BETA0, shape=[n])
    x_sigma_sq     = edmodels.InverseGamma(x_sigma_alpha0, x_sigma_beta0)
    x_sigma        = tf.sqrt(x_sigma_sq)

    x = edmodels.StudentT(df=10.0, loc=x_mu, scale=x_sigma)

    println("Estimating...")

    datadict = PyDict(Dict(x => x0))

    x0_log = log.(input.loaded_samples.x0_values)
    qw_loc_init = vcat(
        mean(x0_log, 1),
        fill(0.0f0, (num_factors - 1, n)))

    if input.inference == :variational

        qw_loc = tf.Variable(qw_loc_init)
        qw_softplus_scale = tf.Variable(tf.fill([num_factors, n], -1.0))
        qw = edmodels.NormalWithSoftplusScale(loc=qw_loc, scale=qw_softplus_scale)

        qx_sigma_sq_mu_param    = tf.Variable(tf.fill([n], -2.0f0), name="qx_sigma_sq_mu_param")
        qx_sigma_sq_sigma_param = tf.Variable(tf.fill([n], -1.0f0), name="qx_sigma_sq_sigma_param")
        qx_sigma_sq = edmodels.TransformedDistribution(
            distribution=edmodels.NormalWithSoftplusScale(qx_sigma_sq_mu_param, qx_sigma_sq_sigma_param),
            bijector=tfdist.bijectors[:Exp](),
            name="LogNormalTransformedDistribution")

        inference = ed.KLqp(Dict(w => qw, x_sigma_sq => qx_sigma_sq),
                            data=datadict)

        optimizer = tf.train[:AdamOptimizer](0.05)
        run_inference(input, inference, 1500, optimizer)

        sess = ed.get_session()

        output_filename = isnull(input.output_filename) ?
            "effects.db" : get(input.output_filename)

        mean_est = sess[:run](qw_loc)
        sigma_est = sess[:run](tf.nn[:softplus](qw_softplus_scale))
        error_sigma = fill(0.0f0, n)
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

    elseif input.inference == :map
        qw_loc = tf.Variable(qw_loc_init)
        qw_map = edmodels.PointMass(params=qw_loc)

        qx_sigma_sq_param = tf.exp(tf.Variable(tf.fill([n], 0.0f0), name="qx_log_sigma_param"))
        qx_sigma_sq_map = edmodels.PointMass(params=qx_sigma_sq_param)

        inference = ed.MAP(Dict(w => qw_map, x_sigma_sq => qx_sigma_sq_map), data=datadict)

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
    else
        error("$(input.inference) inference not implemented for simple linear regression")
    end
end

EXTRUDER_MODELS["simple-linear-regression"] = estimate_simple_linear_regression
