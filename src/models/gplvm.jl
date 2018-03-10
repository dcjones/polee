
function estimate_gplvm(input::ModelInput, num_components::Int=2)
    if input.feature != :transcript
        error("GPLVM only implemented with transcripts")
    end

    sess = ed.get_session()

    num_samples, n = size(input.loaded_samples.x0_values)

    x_mu_bias_mu0 = log(1f0/n)
    x_mu_bias_sigma0 = 2.5f0
    x_mu_bias = edmodels.Normal(loc=tf.fill([1, n], x_mu_bias_mu0),
                              scale=tf.fill([1, n], x_mu_bias_sigma0))

    z = edmodels.Normal(loc=tf.zeros([num_samples, num_components]),
                        scale=tf.fill([num_samples, num_components], 1.0f0))


    # x_mu_gp_scale = tf.cholesky(edutil.rbf(tf.transpose(z)))
    rbfz = edutil.rbf(z) + tf.diag(tf.fill([num_samples], 1.0))
    rbfz = tf_print_span(rbfz, "rbfz span")
    x_mu_gp_scale = tf.cholesky(rbfz)
    # @show sess[:run](x_mu_gp_scale)

    x_mu_gp_scale = tf.tile(tf.expand_dims(x_mu_gp_scale, 0), [n, 1, 1])
    x_mu_gp = edmodels.MultivariateNormalTriL(
        loc=tf.zeros(shape=[n, num_samples]), scale_tril=x_mu_gp_scale)

    x_sigma_alpha0 = tf.constant(SIGMA_ALPHA0, shape=[n])
    x_sigma_beta0 = tf.constant(SIGMA_BETA0, shape=[n])
    x_sigma_sq = edmodels.InverseGamma(x_sigma_alpha0, x_sigma_beta0)
    x_sigma = tf.sqrt(x_sigma_sq)

    x_mu_gp_t = tf.transpose(x_mu_gp, perm=[1, 0])
    x_mu = tf.add(x_mu_gp_t, x_mu_bias)
    # @show x_mu
    # x = edmodels.Normal(loc=x_mu_gp_t, scale=x_sigma)
    # x = edmodels.Normal(loc=x_mu_gp, scale=tf.expand_dims(x_sigma, -1))
    # x_t = tf.transpose(x, perm=tf.constant([1,0]))
    x = edmodels.Normal(loc=x_mu, scale=x_sigma)

    likapprox = RNASeqApproxLikelihood(input, x)

    # inference
    # ---------

    x0_log = tf.log(tf.constant(input.loaded_samples.x0_values))

    qmu_bias_loc = tf.Variable(tf.reduce_mean(x0_log, 0))
    qmu_bias_softplus_scale = tf.Variable(tf.fill([1, n], -1.0))
    qmu_bias = edmodels.NormalWithSoftplusScale(loc=qmu_bias_loc,
                                                scale=qmu_bias_softplus_scale)

    # qz_loc = tf.Variable(tf.multiply(0.001, tf.random_normal([num_samples, num_components])))
    qz_loc = tf.Variable(tf.random_normal([num_samples, num_components]))
    qz_loc = tf_print_span(qz_loc, "qz_loc span")
    qz_softplus_scale = tf.Variable(tf.fill([num_samples, num_components], 0.0))
    qz = edmodels.NormalWithSoftplusScale(loc=qz_loc, scale=qz_softplus_scale)

    qx_mu_gp_loc = tf.Variable(tf.random_normal([n, num_samples]))
    qx_mu_gp_softplus_scale = tf.Variable(tf.random_normal([n, num_samples]))
    qx_mu_gp = edmodels.NormalWithSoftplusScale(loc=qx_mu_gp_loc, scale=qx_mu_gp_softplus_scale)

    qx_sigma_sq_mu_param    = tf.Variable(tf.fill([n], -2.0f0), name="qx_sigma_sq_mu_param")
    qx_sigma_sq_sigma_param = tf.Variable(tf.fill([n], -1.0f0), name="qx_sigma_sq_sigma_param")
    qx_sigma_sq = edmodels.TransformedDistribution(
        distribution=edmodels.NormalWithSoftplusScale(qx_sigma_sq_mu_param, qx_sigma_sq_sigma_param),
        bijector=tfdist.bijectors[:Exp](),
        name="LogNormalTransformedDistribution")

    inference = ed.KLqp(Dict(z => qz, x_mu_gp => qx_mu_gp, x_mu_bias => qmu_bias,
                             x_sigma_sq => qx_sigma_sq),
                        data=Dict(likapprox => Float32[]))

    optimizer = tf.train[:AdamOptimizer](0.05)
    run_inference(input, inference, 1500, optimizer)

    sess = ed.get_session()
    qz_loc_values = @show sess[:run](qz_loc)

    open("gplvm-estimates.csv", "w") do out
        print(out, "sample,")
        println(out, join([string("z", j) for j in 1:num_components], ','))
        for i in 1:num_samples
            print(out, '"', input.loaded_samples.sample_names[i], '"', ',')
            println(out, join(qz_loc_values[i,:], ','))
        end
    end
end


EXTRUDER_MODELS["gplvm"] = estimate_gplvm
