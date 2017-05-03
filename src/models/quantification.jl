

function estimate_quantification(experiment_spec_filename, output_filename)
    # read info from experiment specification
    experiment_spec = YAML.load_file(experiment_spec_filename)
    names = [entry["name"] for entry in experiment_spec]
    filenames = [entry["file"] for entry in experiment_spec]
    sample_factors = [get(entry, "factors", String[]) for entry in experiment_spec]
    num_samples = length(filenames)
    println("Read model specification with ", num_samples, " samples")

    n, likapprox_data, y0 = load_samples(filenames)
    println("Sample data loaded")

    # model
    # -----

    y_mu_mu0 = tf.constant(log(1/n), shape=[n])
    y_mu_sigma0 = tf.constant(10.0, shape=[n])
    y_mu = edmodels.MultivariateNormalDiag(y_mu_mu0, y_mu_sigma0)

    #y_mu_ = tf.Print(y_mu, [tf.reduce_min(y_mu), tf.reduce_max(y_mu)], "y_mu span")
    y_sigma_mu0 = tf.constant(0.0, shape=[n])
    y_sigma_sigma0 = tf.constant(1.0, shape=[n])
    y_log_sigma = edmodels.MultivariateNormalDiag(y_sigma_mu0, y_sigma_sigma0)
    y_sigma = tf.exp(y_log_sigma)
    #y_sigma_ = tf.Print(y_sigma, [tf.reduce_min(y_sigma), tf.reduce_max(y_sigma)], "y_sigma span")

    y_mu_param = tf.matmul(tf.ones([num_samples, 1]),
                           tf.expand_dims(y_mu, 0))

    y_sigma_param = tf.matmul(tf.ones([num_samples, 1]),
                              tf.expand_dims(y_sigma, 0))


    y = edmodels.MultivariateNormalDiag(y_mu_param, y_sigma_param)

    y_sigma_param = tf.matmul(tf.ones([num_samples, 1]),
                              tf.expand_dims(y_sigma, 0))

    likapprox = rnaseq_approx_likelihood.RNASeqApproxLikelihood(
                  y=y, value=likapprox_data)


    # inference
    # ---------

    println("Estimating...")

    qy_mu = tf.Variable(y0)
    #qy_mu = tf.Print(qy_mu,
            #[tf.reduce_min(qy_mu), tf.reduce_mean(qy_mu), tf.reduce_max(qy_mu)],
            #"qy_mu span")
    qy_sigma = tf.identity(tf.Variable(tf.fill([num_samples, n], 0.1)))
    qy = edmodels.MultivariateNormalDiag(qy_mu, qy_sigma)


    qy_mu_mu = tf.Variable(y0[1])
    qy_mu_sigma = tf.identity(tf.Variable(tf.fill([n], 0.1)))
    qy_mu = edmodels.MultivariateNormalDiag(qy_mu_mu, qy_mu_sigma)

    inference = ed.KLqp(PyDict(Dict(y => qy, y_mu => qy_mu)),
                        data=PyDict(Dict(likapprox => likapprox_data)))

    optimizer = tf.train[:AdamOptimizer](1e-2)
    #optimizer = tf.train[:MomentumOptimizer](1e-3, 0.99)
    inference[:run](n_iter=500, optimizer=optimizer)


    sess = ed.get_session()
    return (sess[:run](qy_mu), sess[:run](qy_sigma))
end


EXTRUDER_MODELS["quantification"] = estimate_quantification

