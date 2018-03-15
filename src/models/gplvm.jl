
function estimate_gplvm(input::ModelInput; num_components::Int=2,
                        correct_batch_effects::Bool=false)
    if input.feature != :transcript
        error("GPLVM only implemented with transcripts")
    end

    num_samples, n = size(input.loaded_samples.x0_values)

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

    # batch effects
    # -------------

    x_mu_batch = nothing

    if correct_batch_effects
        w_batch_mu0 = 0.0f0
        w_batch_sigma0 = 2.0f0

        w_batch_sigma = tf.constant(w_batch_sigma0, shape=[num_factors, n])
        w_batch_mu = tf.constant(w_batch_mu0, shape=[num_factors, n])

        w_batch = edmodels.Normal(name="w_batch", loc=w_batch_mu, scale=w_batch_sigma)
        x_mu_batch = tf.matmul(X, w_batch)
    end

    # bias
    # ----

    x_mu_bias_mu0 = log(1f0/n)
    x_mu_bias_sigma0 = 4.0f0
    x_mu_bias = edmodels.Normal(loc=tf.fill([1, n], x_mu_bias_mu0),
                                scale=tf.fill([1, n], x_mu_bias_sigma0))

    # gplvm
    # -----

    z = edmodels.Normal(loc=tf.zeros([num_samples, num_components]),
                        scale=tf.fill([num_samples, num_components], 1.0f0))


    # x_mu_gp_scale = tf.cholesky(edutil.rbf(tf.transpose(z)))
    rbfz = edutil.rbf(z) + tf.diag(tf.fill([num_samples], 1.0f0))
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
    if correct_batch_effects
        x_mu = tf.add(x_mu_batch, x_mu)
    end

    x = edmodels.Normal(loc=x_mu, scale=x_sigma)
    # x = edmodels.StudentT(loc=x_mu, scale=x_sigma)

    likapprox = RNASeqApproxLikelihood(input, x)

    # inference
    # ---------

    x0_log = tf.log(tf.constant(input.loaded_samples.x0_values))

    qx_loc = tf.Variable(x0_log)
    qx_softplus_scale = tf.Variable(tf.fill([num_samples, n], -2.0))
    qx = edmodels.NormalWithSoftplusScale(loc = qx_loc, scale = qx_softplus_scale)

    qmu_bias_loc = tf.Variable(tf.reduce_mean(x0_log, 0))
    qmu_bias_softplus_scale = tf.Variable(tf.fill([1, n], -1.0f0))
    qmu_bias = edmodels.NormalWithSoftplusScale(loc=qmu_bias_loc,
                                                scale=qmu_bias_softplus_scale)

    qz_loc = tf.Variable(tf.random_normal([num_samples, num_components]))
    qz_loc = tf_print_span(qz_loc, "qz_loc span")
    qz_softplus_scale = tf.Variable(tf.fill([num_samples, num_components], 0.0f0))
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

    vars = Dict(z => qz, x => qx, x_mu_gp => qx_mu_gp, x_mu_bias => qmu_bias,
                x_sigma_sq => qx_sigma_sq)

    if correct_batch_effects
        qw_batch_loc = tf.Variable(fill(0.0f0, (num_factors, n)))
        qw_batch_softplus_scale = tf.Variable(tf.fill([num_factors, n], -1.0))
        qw_batch = edmodels.NormalWithSoftplusScale(loc=qw_batch_loc, scale=qw_batch_softplus_scale)

        vars[w_batch] = qw_batch
    end

    inference = ed.KLqp(vars, data=Dict(likapprox => Float32[]))

    optimizer = tf.train[:AdamOptimizer](0.05)
    run_inference(input, inference, 500, optimizer)

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
EXTRUDER_MODELS["batch-gplvm"] = input -> estimate_gplvm(input, correct_batch_effects=true)
