
function estimate_pca(input::ModelInput)
    if input.feature != :transcript
        error("PCA only implemented with transcripts")
    end

    num_samples, n = size(input.loaded_samples.x0_values)
    num_components = 2

    w = edmodels.Normal(loc=tf.zeros([n, num_components]),
                        scale=tf.fill([n, num_components], 1.0f0))
    z = edmodels.Normal(loc=tf.zeros([num_samples, num_components]),
                        scale=tf.fill([num_samples, num_components], 1.0f0))

    w_ = tf_print_span(w, "w span")
    z_ = tf_print_span(z, "z span")
    z_ = tf.Print(z_, [z_[1,1], z_[1,2]], "z 00, 01")

    # x_bias = edmodels.Normal(loc=tf.fill([1, n], log(0.001 * 1/n)), scale=tf.fill([1, n], 5.0f0))
    x_bias = edmodels.Normal(loc=tf.fill([1, n], log(1/n)), scale=tf.fill([1, n], 5.0f0))
    x_bias_ = tf_print_span(x_bias, "x_bias span")

    x = tf.add(x_bias_, tf.matmul(z_, w_, transpose_b=true))
    # x = tf.matmul(z_, w_, transpose_b=true)

    x_err_log_sigma_mu0 = tf.constant(-8.0, shape=[n])
    x_err_log_sigma_sigma0 = tf.constant(1.0, shape=[n])
    x_err_log_sigma = edmodels.MultivariateNormalDiag(x_err_log_sigma_mu0,
                                                      x_err_log_sigma_sigma0)
    x_err_sigma = tf.exp(x_err_log_sigma)
    x_err_sigma = tf_print_span(x_err_sigma, "x_err_sigma span")

    x_err_sigma_param = tf.matmul(tf.ones([num_samples, 1]),
                                  tf.expand_dims(x_err_sigma, 0))
    # x_err_sigma_param = tf.fill([num_samples, n], 0.0001)

    x_err = edmodels.MultivariateNormalDiag(x, x_err_sigma_param)

    likapprox = RNASeqApproxLikelihood(input, x)

    qx_bias_loc = tf.Variable(tf.fill([1,n], log(1/n)))
    qx_bias = edmodels.Normal(loc=qx_bias_loc,
                              scale=tf.nn[:softplus](tf.Variable(tf.zeros([1, n]))))

    qw_loc = tf.Variable(tf.multiply(0.001, tf.random_normal([n, num_components])))
    qw = edmodels.Normal(loc=qw_loc,
                         scale=tf.nn[:softplus](tf.Variable(tf.zeros([n, num_components]))))

    qz_loc = tf.Variable(tf.multiply(0.001, tf.random_normal([num_samples, num_components])))
    qz = edmodels.Normal(loc=qz_loc,
                         scale=tf.nn[:softplus](tf.Variable(tf.zeros([num_samples, num_components]))))

    qx_err_log_sigma_mu_param = tf.Variable(tf.fill([n], 0.0f0), name="qx_err_log_sigma_mu_param")
    qx_err_log_sigma_sigma_param = tf.nn[:softplus](tf.Variable(tf.fill([n], -1.0f0), name="qx_err_log_sigma_sigma_param"))
    qx_err_log_sigma = edmodels.MultivariateNormalDiag(qx_err_log_sigma_mu_param,
                                                       qx_err_log_sigma_sigma_param)

    qx_err_mu_param = tf.Variable(tf.fill([num_samples, n], log(0.1)), name="qx_err_mu_param")
    qx_err_sigma_param = tf.nn[:softplus](tf.Variable(tf.fill([num_samples, n], -1.0f0), name="qx_err_sigma_param"))
    qx_err = edmodels.MultivariateNormalDiag(qx_err_mu_param, qx_err_sigma_param)

    inference = ed.KLqp(Dict(w => qw, z => qz, x_bias => qx_bias,
                             x_err_log_sigma => qx_err_log_sigma, x_err => qx_err),
                        data=Dict(likapprox => Float32[]))

    optimizer = tf.train[:AdamOptimizer](1e-1)
    # inference[:run](n_iter=1000, optimizer=optimizer)
    run_inference(input, inference, 500, optimizer)

    sess = ed.get_session()
    qz_loc_values = @show sess[:run](qz_loc)
    @show extrema(qz_loc_values)
    @show qz_loc_values

    open("estimates.csv", "w") do out
        print(out, "sample,")
        println(out, join([string("pc", j) for j in 1:num_components], ','))
        for i in 1:num_samples
            print(out, '"', input.sample_names[i], '"', ',')
            println(out, join(qz_loc_values[i,:], ','))
        end
    end

    qw_loc_values = sess[:run](qw_loc)
    open("pca-coefficients.csv", "w") do out
        println(out, "component,id,coeff")
        for i in 1:n
            for j in 1:num_components
                @printf(out, "%d,%d,%e\n", j, i, qw_loc_values[i, j])
            end
        end
    end
end



function estimate_batch_pca(input::ModelInput)
    if input.feature != :transcript
        error("PCA only implemented with transcripts")
    end

    num_samples, n = size(input.loaded_samples.x0_values)
    num_components = 2

    # build design matrix
    # -------------------

    factoridx = Dict{String, Int}()
    # factoridx["bias"] = 1
    for factors in input.loaded_samples.sample_factors
        for factor in factors
            get!(factoridx, factor, length(factoridx) + 1)
        end
    end

    num_factors = length(factoridx)
    X_ = zeros(Float32, (num_samples, num_factors))
    for i in 1:num_samples
        for factor in input.loaded_samples.sample_factors[i]
            j = factoridx[factor]
            X_[i, j] = 1
        end
    end
    # X_[:, factoridx["bias"]] = 1
    @show size(X_)
    @show X_
    X = tf.constant(X_)

    @show X[:get_shape]()
    @show (num_factors, n)

    # batch linear regression model specification
    # -------------------------------------------

    # TODO: this notation isn't so good. Maybe something like this to clarify.
    #  y -> x
    #  X -> z

    w_mu0 = 0.0
    w_sigma0 = 1.0
    # w_bias_mu0 = log(1/n)
    w_bias_mu0 = 0.0
    w_bias_sigma0 = 1.0

    x_bias = edmodels.Normal(loc=tf.fill([1, n], 0.0f0), scale=tf.fill([1, n], 5.0f0))

    # w_sigma = tf.concat(
    #               [tf.constant(w_bias_sigma0, shape=[1, n]),
    #                tf.constant(w_sigma0, shape=[num_factors-1, n])], 0)

    # w_mu = tf.concat(
    #               [tf.constant(w_bias_mu0, shape=[1, n]),
    #                tf.constant(w_mu0, shape=[num_factors-1, n])], 0)

    w_sigma = tf.constant(w_sigma0, shape=[num_factors, n])
    w_mu = tf.constant(w_mu0, shape=[num_factors, n])

    w_batch = edmodels.MultivariateNormalDiag(name="w_batch", w_mu, w_sigma)

    #=
    y_sigma_mu0 = tf.constant(0.0, shape=[n])
    y_sigma_sigma0 = tf.constant(1.0, shape=[n])

    y_log_sigma = edmodels.MultivariateNormalDiag(y_sigma_mu0, y_sigma_sigma0)
    y_sigma = tf.exp(y_log_sigma)

    y_sigma_param = tf.matmul(tf.ones([num_samples, 1]),
                              tf.expand_dims(y_sigma, 0))

    X_times_w_batch = tf.matmul(X, w_batch)
    # X_times_w_batch = tf.Print(X_times_w_batch, [tf.reduce_min(X_times_w_batch), tf.reduce_max(X_times_w_batch)], "X*w_batch SPAN")
    y_sigma_param = tf.Print(y_sigma_param, [tf.reduce_min(y_sigma_param), tf.reduce_max(y_sigma_param)], "Y_SIGMA_PARAM")
    x_batch = edmodels.MultivariateNormalDiag(X_times_w_batch,
                                              y_sigma_param)
    # x_batch = edmodels.MultivariateNormalDiag(tf.matmul(X, w_batch),
    #                                           y_sigma_param)
    =#

    x_batch = tf.matmul(X, w_batch)

    # pca model specification
    # -----------------------

    w = edmodels.Normal(loc=tf.zeros([n, num_components]),
                        scale=tf.fill([n, num_components], 1.0f0))
    z = edmodels.Normal(loc=tf.zeros([num_samples, num_components]),
                        scale=tf.fill([num_samples, num_components], 1.0f0))
    # x = tf.transpose(tf.matmul(w, z, transpose_b=true))
    x_pca = tf.matmul(z, w, transpose_b=true)

    # x_batch = tf.Print(x_batch, [tf.reduce_min(x_batch), tf.reduce_max(x_batch)], "X BATCH SPAN", summarize=10)
    # x_batch = tf.Print(x_batch, [x_batch], "X BATCH", summarize=10)
    # x_pca = tf.Print(x_pca, [tf.reduce_min(x_pca), tf.reduce_max(x_pca)], "X PCA", summarize=10)

    x_mu = tf.add(x_bias, tf.add(x_batch, x_pca))

    # x = edmodels.Normal(loc=x_mu, scale=tf.fill([num_samples, n], 0.1f0))
    # x = x_batch

    # x = tf.Print(x, [tf.reduce_min(x), tf.reduce_max(x)], "X", summarize=10)
    # x = x_batch

    likapprox = RNASeqApproxLikelihood(input, x_mu)

    qx_bias_loc = tf.Variable(tf.fill([1,n], log(1/n)))
    qx_bias = edmodels.Normal(loc=qx_bias_loc,
                              scale=tf.nn[:softplus](tf.Variable(tf.zeros([1, n]))))

    # qw_batch_loc = tf.Variable(tf.zeros([num_factors, n]))
    qw_batch_loc = tf.Variable(tf.multiply(0.001, tf.random_normal([num_factors, n])))
    # qw_batch_loc = tf.Print(qw_batch_loc, [tf.reduce_min(qw_batch_loc), tf.reduce_max(qw_batch_loc)], "QW BATCH LOC SPAN")

    # tmp = tf.matmul(X, qw_batch_loc)
    # qw_batch_loc = tf.Print(qw_batch_loc, [tf.reduce_min(tmp), tf.reduce_max(tmp)], "X * QW BATCH LOC SPAN")


    # qw_batch_loc = tf.Print(qw_batch_loc, [qw_batch_loc], "QW BATCH LOC", summarize=10)
    qw_batch_scale = tf.nn[:softplus](tf.Variable(tf.zeros([num_factors, n])))
    # qw_batch_scale = tf.Print(qw_batch_scale, [qw_batch_scale], "QW BATCH SCALE", summarize=10)
    qw_batch = edmodels.Normal(loc=qw_batch_loc, scale=qw_batch_scale)
    # qw_batch = edmodels.Normal(loc=tf.Variable(tf.zeros([num_factors, n])),
    #                            scale=tf.nn[:softplus](tf.Variable(tf.zeros([num_factors, n]))))

    # qw_loc = tf.Variable(tf.zeros([n, num_components]))
    qw_loc = tf.Variable(tf.multiply(0.001, tf.random_normal([n, num_components])))

    # qw_loc = tf.Print(qw_loc, [qw_loc], "QW LOC", summarize=10)
    qw_scale = tf.nn[:softplus](tf.Variable(tf.fill([n, num_components], -5.0)))
    # qw_scale = tf.Print(qw_scale, [qw_scale], "QW SCALE", summarize=10)
    qw = edmodels.Normal(loc=qw_loc, scale=qw_scale)
    # qw = edmodels.Normal(loc=tf.Variable(tf.zeros([n, num_components])),
    #                      scale=tf.nn[:softplus](tf.Variable(tf.zeros([n, num_components]))))

    # qz_loc = tf.Variable(tf.zeros([num_samples, num_components]))
    qz_loc = tf.Variable(tf.multiply(0.001, tf.random_normal([num_samples, num_components])))

    # qz_loc = tf.Print(qz_loc, [qz_loc], "QZ LOC", summarize=10)
    qz_scale = tf.nn[:softplus](tf.Variable(tf.fill([num_samples, num_components], -5.0)))
    # qz_scale = tf.Print(qz_scale, [qz_scale], "QZ SCALE", summarize=10)
    qz = edmodels.Normal(loc=qz_loc, scale=qz_scale)


    # qx_loc = tf.Variable(tf.fill([num_samples,n], log(1/n)))
    # qx = edmodels.Normal(loc=qx_loc,
    #                      scale=tf.nn[:softplus](tf.Variable(tf.zeros([num_samples, n]))))

    # inference = ed.KLqp(Dict(w => qw, w_batch => qw_batch, z => qz, x => qx, x_bias => qx_bias),
    #                     data=Dict(likapprox => Float32[]))

    inference = ed.KLqp(Dict(w => qw, w_batch => qw_batch, z => qz, x_bias => qx_bias),
                        data=Dict(likapprox => Float32[]))

    optimizer = tf.train[:AdamOptimizer](1e-1)
    # optimizer = tf.train[:MomentumOptimizer](1e-9, 0.99)
    run_inference(input, inference, 1000, optimizer)

    sess = ed.get_session()
    qz_loc_values = @show sess[:run](qz_loc)

    open("estimates.csv", "w") do out
        print(out, "sample,")
        println(out, join([string("pc", j) for j in 1:num_components], ','))
        for i in 1:num_samples
            print(out, '"', input.sample_names[i], '"', ',')
            println(out, join(qz_loc_values[i,:], ','))
        end
    end

    qw_loc_values = sess[:run](qw_loc)
    open("pca-coefficients.csv", "w") do out
        println(out, "component,id,transcript_id,coeff")
        for t in input.ts
            i = t.metadata.id
            for j in 1:num_components
                @printf(out, "%d,%d,%s,%e\n", j, i, t.metadata.name, qw_loc_values[i, j])
            end
        end
    end
end

# TODO: I want to compare against PCA on point estimates...

EXTRUDER_MODELS["pca"] = estimate_pca
EXTRUDER_MODELS["batch-pca"] = estimate_batch_pca