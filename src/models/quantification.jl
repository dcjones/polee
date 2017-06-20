
function estimate_quantification(experiment_spec_filename, output_filename,
                                 ts, ts_metadata)
    likapprox_data, y0, sample_factors, sample_names =
            load_samples_from_specification(experiment_spec_filename)

    #sess = ed.get_session()
    #post_mean = sess[:run](tf.nn[:softmax](y0, dim=-1))
    #write_estimates(output_filename, sample_names, post_mean)
    #exit()

    qy_mu_value, qy_sigma_value =
        estimate_quantification(likapprox_data, y0, sample_factors)

    #post_mean = sess[:run](tf.nn[:softmax](qy_mu_value, dim=-1))
    #write_estimates(output_filename, names, post_mean)
end


function estimate_quantification(likapprox_data, y0, sample_factors)
    num_samples = y0[:get_shape]()[1][:value]
    n           = y0[:get_shape]()[2][:value]

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

    qy_mu_param = tf.Variable(y0)
    #qy_mu = tf.Print(qy_mu,
            #[tf.reduce_min(qy_mu), tf.reduce_mean(qy_mu), tf.reduce_max(qy_mu)],
            #"qy_mu span")
    qy_sigma_param = tf.identity(tf.Variable(tf.fill([num_samples, n], 0.1)))
    qy = edmodels.MultivariateNormalDiag(qy_mu_param, qy_sigma_param)


    qy_mu_mu_param = tf.Variable(y0[1])
    qy_mu_sigma_param = tf.identity(tf.Variable(tf.fill([n], 0.1)))
    qy_mu = edmodels.MultivariateNormalDiag(qy_mu_mu_param, qy_mu_sigma_param)

    inference = ed.KLqp(PyDict(Dict(y => qy, y_mu => qy_mu)),
                        data=PyDict(Dict(likapprox => likapprox_data)))

    optimizer = tf.train[:AdamOptimizer](1e-2)
    #optimizer = tf.train[:MomentumOptimizer](1e-3, 0.99)
    inference[:run](n_iter=250, optimizer=optimizer)

    sess = ed.get_session()
    qy_mu_value    = sess[:run](qy_mu_param)
    qy_sigma_value = sess[:run](qy_sigma_param)

    # reset session and graph to free up memory
    tf.reset_default_graph()
    old_sess = ed.get_session()
    old_sess[:close]()
    ed.util[:graphs][:_ED_SESSION] = tf.InteractiveSession()

    return qy_mu_value, qy_sigma_value
end


EXTRUDER_MODELS["quantification"] = estimate_quantification

