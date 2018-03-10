
function estimate_pca(input::ModelInput; num_components::Int=2,
                      correct_batch_effects::Bool=false)

    if input.feature != :transcript
        error("PCA only implemented with transcripts")
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

    w_mu0 = 0.0f0
    w_sigma0 = 2.0f0

    w_sigma = tf.constant(w_sigma0, shape=[num_factors, n])
    w_mu = tf.constant(w_mu0, shape=[num_factors, n])

    w_batch = edmodels.Normal(name="w_batch", loc=w_mu, scale=w_sigma)
    mu_batch = tf.matmul(X, w_batch)

    # bias
    # ----

    w_bias_mu0 = log(1f0/n)
    w_bias_sigma0 = 4.0f0

    mu_bias = edmodels.Normal(loc=tf.fill([1, n], w_bias_mu0),
                              scale=tf.fill([1, n], w_bias_sigma0))

    # pca model specification
    # -----------------------

    w = edmodels.Normal(loc=tf.zeros([n, num_components]),
                        scale=tf.fill([n, num_components], 1.0f0))
    z = edmodels.Normal(loc=tf.zeros([num_samples, num_components]),
                        scale=tf.fill([num_samples, num_components], 1.0f0))

    mu_pca = tf.matmul(z, w, transpose_b=true)

    x_mu = tf.add(mu_bias, mu_pca)

    if correct_batch_effects
        x_mu = tf.add(mu_batch, x_mu)
    end

    x_sigma_alpha0 = tf.constant(SIGMA_ALPHA0, shape=[n])
    x_sigma_beta0 = tf.constant(SIGMA_BETA0, shape=[n])
    x_sigma_sq = edmodels.InverseGamma(x_sigma_alpha0, x_sigma_beta0)
    x_sigma = tf.sqrt(x_sigma_sq)
    # x = edmodels.StudentT(df=10.0f0, loc=x_mu, scale=x_sigma)
    x = edmodels.Normal(loc=x_mu, scale=x_sigma)

    likapprox = RNASeqApproxLikelihood(input, x)

    # inference
    # ---------

    x0_log = tf.log(tf.constant(input.loaded_samples.x0_values))

    qmu_bias_loc = tf.Variable(tf.reduce_mean(x0_log, 0))
    qmu_bias_softplus_scale = tf.Variable(tf.fill([1, n], -1.0f0))
    qmu_bias = edmodels.NormalWithSoftplusScale(loc=qmu_bias_loc,
                                                scale=qmu_bias_softplus_scale)

    qw_loc = tf.Variable(tf.multiply(0.001f0, tf.random_normal([n, num_components])))
    qw_softplus_scale = tf.Variable(tf.fill([n, num_components], -2.0f0))
    qw = edmodels.NormalWithSoftplusScale(loc=qw_loc, scale=qw_softplus_scale)

    qz_loc = tf.Variable(tf.multiply(0.001f0, tf.random_normal([num_samples, num_components])))
    qz_softplus_scale = tf.Variable(tf.fill([num_samples, num_components], -2.0f0))
    qz = edmodels.NormalWithSoftplusScale(loc=qz_loc, scale=qz_softplus_scale)

    qx_sigma_sq_mu_param    = tf.Variable(tf.fill([n], -2.0f0), name="qx_sigma_sq_mu_param")
    qx_sigma_sq_sigma_param = tf.Variable(tf.fill([n], -1.0f0), name="qx_sigma_sq_sigma_param")
    qx_sigma_sq = edmodels.TransformedDistribution(
        distribution=edmodels.NormalWithSoftplusScale(qx_sigma_sq_mu_param, qx_sigma_sq_sigma_param),
        bijector=tfdist.bijectors[:Exp](),
        name="LogNormalTransformedDistribution")

    vars = Dict(w => qw, z => qz, mu_bias => qmu_bias,
                x_sigma_sq => qx_sigma_sq)

    if correct_batch_effects
        qw_batch_loc = tf.Variable(fill(0.0f0, (num_factors, n)))
        qw_batch_softplus_scale = tf.Variable(tf.fill([num_factors, n], -1.0f0))
        qw_batch = edmodels.NormalWithSoftplusScale(loc=qw_batch_loc, scale=qw_batch_softplus_scale)

        vars[w_batch] = qw_batch
    end

    inference = ed.KLqp(vars, data=Dict(likapprox => Float32[]))

    optimizer = tf.train[:AdamOptimizer](0.05)
    run_inference(input, inference, 1500, optimizer)

    sess = ed.get_session()
    qz_loc_values = sess[:run](qz_loc)

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
EXTRUDER_MODELS["batch-pca"] = input -> estimate_pca(input, correct_batch_effects=true)