
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
    w_sigma0 = 0.1
    w_bias_sigma0 = 100.0

    w_sigma = tf.concat(
                  [tf.constant(w_bias_sigma0, shape=[1, n]),
                   tf.constant(w_sigma0, shape=[num_factors-1, n])], 0)
    W = edmodels.MultivariateNormalDiag(
            name="W", tf.constant(0.0, shape=[num_factors, n]), w_sigma)

    y = tf.matmul(X, W)

    musigma = rnaseq_approx_likelihood.RNASeqApproxLikelihood(
                y=y, value=musigma_data)

    # inference
    # ---------

    println("Estimating...")
    sess = ed.get_session()
    datadict = PyDict(Dict(musigma => musigma_data))

    map_iterations = 10
    qw_map_param = tf.Variable(tf.fill([num_factors, n], 5.0))
    qw_map = edmodels.PointMass(params=qw_map_param)
    inference = ed.MAP(Dict(W => qw_map), data=datadict)

    #inference[:run](n_iter=map_iterations)

    #optimizer = tf.train[:MomentumOptimizer](1e-7, 0.9)
    optimizer = tf.train[:AdamOptimizer](0.5)
    inference[:run](n_iter=map_iterations, optimizer=optimizer)


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

    qw_mu = qw_map_param

    @time write_effects(output_filename, factoridx, sess[:run](qw_mu))
end


EXTRUDER_MODELS["linear-regression"] = estimate_linear_regression


