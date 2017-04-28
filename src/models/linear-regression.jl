
function estimate_linear_regression(experiment_spec_filename, output_filename)
    # read info from experiment specification
    experiment_spec = YAML.load_file(experiment_spec_filename)
    names = [entry["name"] for entry in experiment_spec]
    filenames = [entry["file"] for entry in experiment_spec]
    sample_factors = [get(entry, "factors", String[]) for entry in experiment_spec]
    num_samples = length(filenames)
    println("Read model specification with ", num_samples, " samples")

    # build design matrix
    # -------------------

    factoridx = Dict{String, Int}()
    factoridx["bias"] = 1
    for factors in sample_factors
        for factor in factors
            get!(factoridx, factor, length(factoridx) + 1)
        end
    end

    num_factors = length(factoridx)
    X_ = zeros(Float32, (num_samples, num_factors))
    for i in 1:num_samples
        for factor in sample_factors[i]
            j = factoridx[factor]
            X_[i, j] = 1
        end
    end
    X_[:, factoridx["bias"]] = 1
    X = tf.constant(X_)

    n, musigma_data, y0 = load_samples(filenames)
    println("Sample data loaded")

    # model specification
    # -------------------

    # TODO: pass these in as parameters
    w_mu0 = 0.0
    w_sigma0 = 1.0
    w_bias_mu0 = log(1/n)
    w_bias_sigma0 = 10.0

    w_sigma = tf.concat(
                  [tf.constant(w_bias_sigma0, shape=[1, n]),
                   tf.constant(w_sigma0, shape=[num_factors-1, n])], 0)
    w_mu = tf.concat(
                  [tf.constant(w_bias_mu0, shape=[1, n]),
                   tf.constant(w_mu0, shape=[num_factors-1, n])], 0)

    W = edmodels.MultivariateNormalDiag(name="W", w_mu, w_sigma)

    mu = tf.matmul(X, W)

    #y_sigma_alpha = tf.constant(1.0, shape=[n])
    #y_sigma_beta = tf.constant(1.0, shape=[n])
    #y_sigma = edmodels.InverseGamma(y_sigma_alpha, y_sigma_beta)

    y_sigma_mu0 = tf.constant(0.0, shape=[n])
    y_sigma_sigma0 = tf.constant(1.0, shape=[n])
    #y_sigma = edmodels.TransformedDistribution(
                #distribution=tfdist.MultivariateNormalDiag(y_sigma_mu0,
                                                           #y_sigma_sigma0),
                #bijector=tfdist.bijector[:Exp]())

    y_log_sigma = edmodels.MultivariateNormalDiag(y_sigma_mu0, y_sigma_sigma0)
    y_sigma = tf.exp(y_log_sigma)

    y_sigma_param = tf.matmul(tf.ones([num_samples, 1]),
                              tf.expand_dims(y_sigma, 0))

    y = edmodels.MultivariateNormalDiag(mu, y_sigma_param)

    musigma = rnaseq_approx_likelihood.RNASeqApproxLikelihood(
                y=y, value=musigma_data)

    # inference
    # ---------

    println("Estimating...")

    # optimize against point estimates
    init_iterations = 250

    #qw_init_param = tf.Variable(tf.fill([num_factors, n], 0.0))
    qw_init_param = tf.Variable(w_mu)
    qw = edmodels.PointMass(params=qw_init_param)

    qy_log_sigma_init_param = tf.Variable(tf.fill([n], 0.0))
    qy_log_sigma = edmodels.PointMass(params=qy_log_sigma_init_param)

    inference = ed.MAP(Dict(W => qw, y_log_sigma => qy_log_sigma),
                       data=PyDict(Dict(y => y0)))

    optimizer = tf.train[:AdamOptimizer](0.1)
    inference[:run](n_iter=init_iterations, optimizer=optimizer)


    # optimize over the full model
    sess = ed.get_session()

    map_iterations = 500

    #qw_param = tf.Variable(tf.fill([num_factors, n], 0.0))
    qw_param = tf.Variable(sess[:run](qw_init_param))
    #qw_param = tf.Print(qw_param, [tf.reduce_min(qw_param),
                                   #tf.reduce_max(qw_param)], "W span")
    qw = edmodels.PointMass(params=qw_param)

    qy_log_sigma_param = tf.Variable(sess[:run](qy_log_sigma_init_param))
    #qy_log_sigma_param = tf.Print(qy_log_sigma_param,
                              #[tf.reduce_min(qy_log_sigma_param),
                               #tf.reduce_max(qy_log_sigma_param)], "sigma span")
    qy_log_sigma = edmodels.PointMass(params=qy_log_sigma_param)

    inference = ed.MAP(Dict(W => qw, y_log_sigma => qy_log_sigma),
                       data=PyDict(Dict(musigma => musigma_data)))

    #inference = ed.MAP(Dict(W => qw), data=datadict)

    #inference[:run](n_iter=map_iterations)

    #optimizer = tf.train[:MomentumOptimizer](1e-12, 0.99)
    optimizer = tf.train[:AdamOptimizer](0.1)
    inference[:run](n_iter=map_iterations, optimizer=optimizer)
    #inference[:run](n_iter=map_iterations)


    #=
    vi_iterations = 500
    qw_mu = tf.Variable(sess[:run](qw_map_param))
    qw_sigma = tf.identity(tf.Variable(tf.fill([num_factors, n], 1.0)))
    qw = edmodels.MultivariateNormalDiag(name="qw", qw_mu, qw_sigma)

    inference = ed.KLqp(Dict(W => qw), data=datadict)

    learning_rate = 1e-3
    beta1 = 0.7
    beta2 = 0.99
    optimizer = tf.train[:AdamOptimizer](learning_rate, beta1, beta2)
    inference[:run](n_iter=vi_iterations, optimizer=optimizer)
    #inference[:run](n_iter=vi_iterations)
    =#

    @time write_effects(output_filename, factoridx,
                        sess[:run](qw_param),
                        sess[:run](tf.exp(qy_log_sigma_param)))
end


EXTRUDER_MODELS["linear-regression"] = estimate_linear_regression


