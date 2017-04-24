
# This is obviously not very DRY, since nearly all of it is from
# linear-regression, but I don't really give a shit about this model for
# anything other than comparison.

function estimate_simple_linear_regression(experiment_spec_filename,
                                           output_filename)
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
    @show n
    println("Sample data loaded")

    # model specification
    # -------------------

    # TODO: pass these in as parameters
    w_mu0 = 0.0
    w_sigma0 = 1.0
    w_bias_sigma0 = 100.0

    w_sigma = tf.concat(
                  [tf.constant(w_bias_sigma0, shape=[1, n]),
                   tf.constant(w_sigma0, shape=[num_factors-1, n])], 0)
    W = edmodels.MultivariateNormalDiag(
            name="W", tf.constant(0.0, shape=[num_factors, n]), w_sigma)

    mu = tf.matmul(X, W)

    y_sigma_alpha = tf.constant(1.0, shape=[n])
    y_sigma_beta = tf.constant(1.0, shape=[n])
    y_sigma = edmodels.InverseGamma(y_sigma_alpha, y_sigma_beta)

    y_sigma_param = tf.matmul(tf.ones([num_samples, 1]),
                              tf.expand_dims(y_sigma, 0))

    y = edmodels.MultivariateNormalDiag(mu, y_sigma_param)

    # inference
    # ---------

    println("Estimating...")
    sess = ed.get_session()
    datadict = PyDict(Dict(y => y0))

    map_iterations = 2500
    qw_map_param = tf.Variable(tf.fill([num_factors, n], 0.0))
    qw_map = edmodels.PointMass(params=qw_map_param)

    qy_sigma_param = tf.Variable(tf.fill([n], 1.0))
    qy_sigma = edmodels.PointMass(params=qy_sigma_param)

    inference = ed.MAP(Dict(W => qw_map, y_sigma => qy_sigma), data=datadict)

    #inference[:run](n_iter=map_iterations)

    #optimizer = tf.train[:MomentumOptimizer](1e-7, 0.9)
    optimizer = tf.train[:AdamOptimizer](0.1)
    inference[:run](n_iter=map_iterations, optimizer=optimizer)

    qw_mu = qw_map_param

    @time write_effects(output_filename, factoridx,
                        sess[:run](qw_mu), sess[:run](qy_sigma_param))
end


EXTRUDER_MODELS["simple-linear-regression"] = estimate_simple_linear_regression
