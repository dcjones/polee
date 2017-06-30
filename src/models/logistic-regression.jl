
# This does logistic categorical regression on provided categories

function estimate_logistic_regression(input::ModelInput)
    if input.feature == :transcript
        qy_mu_value, qy_sigma_value =
            estimate_transcript_expression(input)
    elseif input.feature == :gene
        qy_mu_value, qy_sigma_value =
            estimate_gene_expression(input)
    else
        error("Logistic regression for $(feature)s not supported")
    end

    # build design matrix
    # -------------------

    tic()
    num_samples = length(input.sample_names)
    sample_factors = input.sample_factors
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

    likapprox_train = tf.gather(likapprox_data, idx_train)
    likapprox_test = tf.gather(likapprox_data, idx_test)

    # quantification
    # --------------

    qy_mu_train    = tf.gather(qy_mu, idx_train)
    qy_mu_test     = tf.gather(qy_mu, idx_test)

    qy_sigma_train = tf.gather(qy_sigma, idx_train)
    qy_sigma_test  = tf.gather(qy_sigma, idx_test)

    # model specification
    # -------------------

    w_mu0 = 0.0
    w_sigma0 = 1.0

    w_sigma = tf.constant(w_sigma0, shape=[n, num_factors])
    w_mu = tf.constant(w_mu0, shape=[n, num_factors])
    W = edmodels.MultivariateNormalDiag(name="W", w_mu, w_sigma)
    #W_ = tf.Print(W, [tf.reduce_min(W), tf.reduce_max(W)], "W span")
    W_ = W

    #b = edmodels.MultivariateNormalDiag(name="b",
                                        #tf.constant(w_mu0, shape=[num_factors]),
                                        #tf.constant(w_sigma0, shape=[num_factors]))

    y = edmodels.MultivariateNormalDiag(name="y", qy_mu_train, qy_sigma_train)
    #y_ = tf.Print(y, [tf.reduce_min(y), tf.reduce_max(y)], "y span")
    y_ = y

    # TODO: Ahhhh, WTF, this doesn't even work with point estimates it looks
    # like. Something is wrong here.

    yw = tf.matmul(qy_mu_train, W_)
    #yw = tf.Print(yw, [tf.reduce_min(yw), tf.reduce_max(yw)], "yw span")
    p = tf.nn[:softmax](yw)
    #p = tf.nn[:softmax](tf.matmul(qy_mu_train, W_))
    #p = tf.nn[:softmax](tf.add(tf.matmul(y_, W_), b))

    #ywexp = tf.exp(tf.matmul(y_, W_))
    #p = tf.divide(ywexp, tf.add(tf.reduce_sum(ywexp), 1.0))

    #p = tf.Print(p, [tf.reduce_min(p), tf.reduce_max(p)], "p span")
    p = tf.Print(p, [ed.criticisms[:categorical_accuracy](X_train_true, p)],
                 "train accuracy")

    p = tf.Print(p,
                 [ed.criticisms[:categorical_accuracy](
                    X_test_true,
                    tf.nn[:softmax](tf.matmul(qy_mu_test, W_)))],
                 "test accuracy")
    
    X = edmodels.Categorical(p=p)

    # inference
    # ---------

    println("Estimating...")
    sess = ed.get_session()

    vi_iterations = 2000
    qw_mu = tf.Variable(w_mu)
    qw_sigma = tf.identity(tf.Variable(tf.fill([n, num_factors], -15.0)))
    qw = edmodels.MultivariateNormalDiag(qw_mu, tf.exp(qw_sigma))

    inference = ed.KLqp(PyDict(Dict(W => qw)),
                        data=PyDict(Dict(X => X_train)))


    #qw_param = tf.Variable(w_mu)
    #qw_map = edmodels.PointMass(params=qw_param)
    #inference = ed.MAP(PyDict(Dict(W => qw_map)),
                       #data=PyDict(Dict(X => X_train)))

    optimizer = tf.train[:MomentumOptimizer](1e-9, 0.999)
    #optimizer = tf.train[:AdamOptimizer](1e-6)
    inference[:run](n_iter=vi_iterations,
                    optimizer=optimizer,
                    logdir="/home/dcjones/prj/extruder/testing/e-geuv-1/logs")

    # evaluate
    # --------

    y_test = edmodels.MultivariateNormalDiag(name="y", qy_mu_test, qy_sigma_test)
    X_true = tf.one_hot(X_test, num_factors)
    eval_iterations = 100
    avg_accuracy = 0.0

    y_sample = y_test[:sample]()
    w_sample = qw[:sample]()
    p_pred = tf.nn[:softmax](tf.matmul(y_sample, w_sample))
    accuracy = ed.criticisms[:categorical_accuracy](X_true, p_pred)

    for _ in 1:eval_iterations
        a = sess[:run](accuracy)
        @show a
        avg_accuracy += a
    end
    avg_accuracy /= eval_iterations
    @show avg_accuracy
end


EXTRUDER_MODELS["logistic-regression"] = estimate_logistic_regression

