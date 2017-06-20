
# This does logistic categorical regression on provided categories

function estimate_simple_logistic_regression(experiment_spec_filename, output_filename,
                                             ts, ts_metadata)
    # read info from experiment specification
    experiment_spec = YAML.load_file(experiment_spec_filename)
    names = [entry["name"] for entry in experiment_spec]
    filenames = [entry["file"] for entry in experiment_spec]
    sample_factors = [get(entry, "factors", String[]) for entry in experiment_spec]
    num_samples = length(filenames)
    println("Read model specification with ", num_samples, " samples")

    # build design matrix
    # -------------------

    tic()
    factoridx = Dict{String, Int}()
    for factors in sample_factors
        for factor in factors
            get!(factoridx, factor, length(factoridx) + 1)
        end
    end

    num_factors = length(factoridx)
    Xobs_ = Array(Int32, num_samples)
    for i in 1:num_samples
        for factor in sample_factors[i]
            j = factoridx[factor]
            Xobs_[i] = j - 1
        end
    end
    Xobs = tf.constant(Xobs_)
    toc()

    n, musigma_data, y0 = load_samples(filenames, ts_metadata)
    println("Sample data loaded")

    # pertition into training and testing sets
    idx = shuffle(0:num_samples-1)
    k = floor(Int, num_samples * 0.75)
    idx_train = idx[1:k]
    idx_test  = idx[k+1:end]

    X_train = tf.gather(Xobs, idx_train)
    X_test  = tf.gather(Xobs, idx_test)

    X_train_true = tf.one_hot(X_train, num_factors)
    X_test_true  = tf.one_hot(X_test, num_factors)

    y0_train = tf.gather(y0, idx_train)
    y0_test  = tf.gather(y0, idx_test)

    # model specification
    # -------------------

    w_mu0 = 0.0
    w_sigma0 = 1.0

    w_sigma = tf.constant(w_sigma0, shape=[n, num_factors])
    w_mu = tf.constant(w_mu0, shape=[n, num_factors])
    W = edmodels.MultivariateNormalDiag(name="W", w_mu, w_sigma)

    p = tf.nn[:softmax](tf.matmul(y0_train, W))
    #p = tf.Print(p, [tf.reduce_min(p), tf.reduce_max(p)], "p span")

    #p = tf.Print(p, [ed.criticisms[:categorical_accuracy](X_train_true, p)],
                 #"train accuracy")

    #p = tf.Print(p,
                 #[ed.criticisms[:categorical_accuracy](
                    #X_test_true,
                    #tf.nn[:softmax](tf.matmul(y0_test, W)))],
                 #"test accuracy")

    X = edmodels.Categorical(p=p)

    # inference
    # ---------

    println("Estimating...")
    sess = ed.get_session()

    map_iterations = 2500
    qw_param = tf.Variable(w_mu)
    qw_map = edmodels.PointMass(params=qw_param)

    inference = ed.MAP(Dict(W => qw_map), data=PyDict(Dict(X => X_train)))

    optimizer = tf.train[:AdamOptimizer](1e-6)
    inference[:run](n_iter=map_iterations, optimizer=optimizer)

    # evaluate
    # --------

    p_pred = ed.copy(p, PyDict(Dict(y0_train => y0_test, W => qw_param)))
    X_true = tf.one_hot(X_test, num_factors)
    @show sess[:run](ed.criticisms[:categorical_accuracy](X_true, p_pred))
end


EXTRUDER_MODELS["simple-logistic-regression"] = estimate_simple_logistic_regression

