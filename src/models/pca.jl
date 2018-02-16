
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

    mu_bias = edmodels.Normal(loc=tf.fill([1, n], log(1/n)), scale=tf.fill([1, n], 2.5f0))

    x_mu = tf.add(mu_bias, tf.matmul(z, w, transpose_b=true))

    x_sigma_alpha0 = tf.constant(SIGMA_ALPHA0, shape=[n])
    x_sigma_beta0 = tf.constant(SIGMA_BETA0, shape=[n])
    x_sigma_sq = edmodels.InverseGamma(x_sigma_alpha0, x_sigma_beta0)
    x_sigma = tf.sqrt(x_sigma_sq)

    x = edmodels.StudentT(df=10.0, loc=x_mu, scale=x_sigma)

    likapprox = RNASeqApproxLikelihood(input, x)

    x0_log = tf.log(tf.constant(input.loaded_samples.x0_values))

    qmu_bias_loc = tf.Variable(tf.reduce_mean(x0_log, 0))
    qmu_bias = edmodels.Normal(loc=qmu_bias_loc,
                               scale=tf.nn[:softplus](tf.Variable(tf.zeros([1, n]))))

    qw_loc = tf.Variable(tf.multiply(0.001, tf.random_normal([n, num_components])))
    qw = edmodels.Normal(loc=qw_loc,
                         scale=tf.nn[:softplus](tf.Variable(tf.zeros([n, num_components]))))

    qz_loc = tf.Variable(tf.multiply(0.001, tf.random_normal([num_samples, num_components])))
    qz = edmodels.Normal(loc=qz_loc,
                         scale=tf.nn[:softplus](tf.Variable(tf.zeros([num_samples, num_components]))))

    qx_sigma_sq_mu_param    = tf.Variable(tf.fill([n], 0.0f0), name="qx_sigma_sq_mu_param")
    qx_sigma_sq_sigma_param = tf.Variable(tf.fill([n], 1.0f0), name="qx_sigma_sq_sigma_param")
    qx_sigma_sq = edmodels.TransformedDistribution(
        distribution=edmodels.NormalWithSoftplusScale(qx_sigma_sq_mu_param, qx_sigma_sq_sigma_param),
        bijector=tfdist.bijectors[:Exp](),
        name="LogNormalTransformedDistribution")

    qx_mu_param = tf.Variable(x0_log, name="qx_mu_param")
    qx_sigma_param = tf.nn[:softplus](tf.Variable(tf.fill([num_samples, n], -1.0f0), name="qx_sigma_param"))
    qx = edmodels.MultivariateNormalDiag(qx_mu_param, qx_sigma_param)

    inference = ed.KLqp(Dict(w => qw, z => qz, mu_bias => qmu_bias,
                             x_sigma_sq => qx_sigma_sq, x => qx),
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
            print(out, '"', input.loaded_samples.sample_names[i], '"', ',')
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
    X = tf.constant(X_)


    # batch linear regression model specification
    # -------------------------------------------

    w_mu0 = 0.0
    w_sigma0 = 1.0
    # w_bias_mu0 = log(1/n)
    w_bias_mu0 = 0.0
    w_bias_sigma0 = 1.0

    mu_bias = edmodels.Normal(loc=tf.fill([1, n], log(1/n)), scale=tf.fill([1, n], 2.5f0))

    w_sigma = tf.constant(w_sigma0, shape=[num_factors, n])
    w_mu = tf.constant(w_mu0, shape=[num_factors, n])

    w_batch = edmodels.MultivariateNormalDiag(name="w_batch", w_mu, w_sigma)

    mu_batch = tf.matmul(X, w_batch)

    # pca model specification
    # -----------------------

    w = edmodels.Normal(loc=tf.zeros([n, num_components]),
                        scale=tf.fill([n, num_components], 1.0f0))
    z = edmodels.Normal(loc=tf.zeros([num_samples, num_components]),
                        scale=tf.fill([num_samples, num_components], 1.0f0))
    mu_pca = tf.matmul(z, w, transpose_b=true)

    x_mu = tf.add(mu_bias, tf.add(mu_batch, mu_pca))

    x_sigma_alpha0 = tf.constant(SIGMA_ALPHA0, shape=[n])
    x_sigma_beta0 = tf.constant(SIGMA_BETA0, shape=[n])
    x_sigma_sq = edmodels.InverseGamma(x_sigma_alpha0, x_sigma_beta0)
    x_sigma = tf.sqrt(x_sigma_sq)
    x = edmodels.StudentT(df=10.0, loc=x_mu, scale=x_sigma)

    likapprox = RNASeqApproxLikelihood(input, x)

    # inference
    # ---------

    x0_log = tf.log(tf.constant(input.loaded_samples.x0_values))

    qmu_bias_loc = tf.Variable(tf.reduce_mean(x0_log, 0))
    qmu_bias = edmodels.Normal(loc=qmu_bias_loc,
                               scale=tf.nn[:softplus](tf.Variable(tf.zeros([1, n]))))

    qw_batch_loc = tf.Variable(tf.multiply(0.001, tf.random_normal([num_factors, n])))


    qw_batch_scale = tf.nn[:softplus](tf.Variable(tf.zeros([num_factors, n])))
    qw_batch = edmodels.Normal(loc=qw_batch_loc, scale=qw_batch_scale)

    qw_loc = tf.Variable(tf.multiply(0.001, tf.random_normal([n, num_components])))

    qw_scale = tf.nn[:softplus](tf.Variable(tf.fill([n, num_components], -5.0)))
    qw = edmodels.Normal(loc=qw_loc, scale=qw_scale)

    qz_loc = tf.Variable(tf.multiply(0.001, tf.random_normal([num_samples, num_components])))

    qz_scale = tf.nn[:softplus](tf.Variable(tf.fill([num_samples, num_components], -5.0)))
    qz = edmodels.Normal(loc=qz_loc, scale=qz_scale)

    qx_sigma_sq_mu_param    = tf.Variable(tf.fill([n], 0.0f0), name="qx_sigma_sq_mu_param")
    qx_sigma_sq_sigma_param = tf.Variable(tf.fill([n], 1.0f0), name="qx_sigma_sq_sigma_param")
    qx_sigma_sq = edmodels.TransformedDistribution(
        distribution=edmodels.NormalWithSoftplusScale(qx_sigma_sq_mu_param, qx_sigma_sq_sigma_param),
        bijector=tfdist.bijectors[:Exp](),
        name="LogNormalTransformedDistribution")

    qx_mu_param = tf.Variable(x0_log, name="qx_mu_param")
    qx_sigma_param = tf.nn[:softplus](tf.Variable(tf.fill([num_samples, n], -1.0f0), name="qx_sigma_param"))
    qx = edmodels.MultivariateNormalDiag(qx_mu_param, qx_sigma_param)

    inference = ed.KLqp(Dict(w => qw, w_batch => qw_batch, z => qz, mu_bias => qmu_bias,
                             x_sigma_sq => qx_sigma_sq, x => qx),
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
            print(out, '"', input.loaded_samples.sample_names[i], '"', ',')
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