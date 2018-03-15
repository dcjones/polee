

function estimate_pca(input::ModelInput; num_components::Int=2,
                      correct_batch_effects::Bool=false)

    if input.feature == :transcript
        estimate_transcript_pca(input, num_components=num_components,
                                correct_batch_effects=correct_batch_effects)
    elseif input.feature == :splicing
        estimate_splicing_pca(input, num_components=num_components,
                              correct_batch_effects=correct_batch_effects)
    else
        error("$(input.feature) feature not supported by pca")
    end
end


function estimate_transcript_pca(input::ModelInput; num_components::Int=8,
                                 correct_batch_effects::Bool=false)


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

    qx_loc = tf.Variable(x0_log, name="qx_loc")
    qx_softplus_scale = tf.Variable(tf.fill([num_samples, n], -2.0), name="qx_softplus_scale")
    qx = edmodels.NormalWithSoftplusScale(loc = qx_loc, scale = qx_softplus_scale)

    qmu_bias_loc = tf.Variable(tf.reduce_mean(x0_log, 0), name="qmu_bias_loc")
    qmu_bias_softplus_scale = tf.Variable(tf.fill([1, n], -1.0f0), name="qmu_bias_softplus_scale")
    qmu_bias = edmodels.NormalWithSoftplusScale(loc=qmu_bias_loc,
                                                scale=qmu_bias_softplus_scale)

    qw_loc = tf.Variable(tf.multiply(0.001f0, tf.random_normal([n, num_components])), name="qw_loc")
    qw_softplus_scale = tf.Variable(tf.fill([n, num_components], -2.0f0), name="qw_softplus_scale")
    qw = edmodels.NormalWithSoftplusScale(loc=qw_loc, scale=qw_softplus_scale)

    qz_loc = tf.Variable(tf.multiply(0.001f0, tf.random_normal([num_samples, num_components])), name="qz_loc")
    qz_softplus_scale = tf.Variable(tf.fill([num_samples, num_components], -2.0f0), name="qz_softplus_scale")
    qz = edmodels.NormalWithSoftplusScale(loc=qz_loc, scale=qz_softplus_scale)

    qx_sigma_sq_mu_param    = tf.Variable(tf.fill([n], -2.0f0), name="qx_sigma_sq_mu_param")
    qx_sigma_sq_sigma_param = tf.Variable(tf.fill([n], -1.0f0), name="qx_sigma_sq_sigma_param")
    qx_sigma_sq = edmodels.TransformedDistribution(
        distribution=edmodels.NormalWithSoftplusScale(qx_sigma_sq_mu_param, qx_sigma_sq_sigma_param),
        bijector=tfdist.bijectors[:Exp](),
        name="LogNormalTransformedDistribution")

    vars = Dict(x => qx, w => qw, z => qz, mu_bias => qmu_bias,
                x_sigma_sq => qx_sigma_sq)

    if correct_batch_effects
        qw_batch_loc = tf.Variable(fill(0.0f0, (num_factors, n)), name="qw_batch_loc")
        qw_batch_softplus_scale = tf.Variable(tf.fill([num_factors, n], -1.0f0), name="qw_batch_softplus_scale")
        qw_batch = edmodels.NormalWithSoftplusScale(loc=qw_batch_loc, scale=qw_batch_softplus_scale)

        vars[w_batch] = qw_batch
    end

    inference = ed.KLqp(vars, data=Dict(likapprox => Float32[]))

    optimizer = tf.train[:AdamOptimizer](0.05)
    run_inference(input, inference, 1000, optimizer)

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


function estimate_splicing_pca(input::ModelInput; num_components::Int=2,
                               correct_batch_effects::Bool=false)
    num_samples, n = size(input.loaded_samples.x0_values)

    (num_features,
     feature_idxs, feature_transcript_idxs,
     antifeature_idxs, antifeature_transcript_idxs) = splicing_features(input)

    @show num_features

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

    # TODO:

    # bias
    # ----

    w_bias_mu0 = 0.0f0
    w_bias_sigma0 = 10.0f0

    mu_bias = edmodels.Normal(loc=tf.fill([1, num_features], w_bias_mu0),
                              scale=tf.fill([1, num_features], w_bias_sigma0))

    # pca model specification
    # -----------------------

    w = edmodels.Normal(loc=tf.zeros([num_features, num_components]),
                        scale=tf.fill([num_features, num_components], 1.0f0))
    z = edmodels.Normal(loc=tf.zeros([num_samples, num_components]),
                        scale=tf.fill([num_samples, num_components], 1.0f0))

    # w_ = tf_print_span(w, "w span")
    # z_ = tf_print_span(z, "z span")

    mu_pca = tf.matmul(z, w, transpose_b=true)

    mu_pca = tf_print_span(mu_pca, "mu_pca span")

    # mu_bias_ = tf_print_span(mu_bias, "mu_bias span")

    x_mu = tf.add(mu_bias, mu_pca)
    # x_mu = tf_print_span(x_mu, "x_mu span")

    # TODO:
    # if correct_batch_effects
    #     x_mu = tf.add(mu_batch, x_mu)
    # end

    x_sigma_alpha0 = tf.constant(SIGMA_ALPHA0, shape=[num_features])
    x_sigma_beta0 = tf.constant(SIGMA_BETA0, shape=[num_features])
    x_sigma_sq = edmodels.InverseGamma(x_sigma_alpha0, x_sigma_beta0)
    x_sigma = tf.sqrt(x_sigma_sq)
    x = edmodels.Normal(loc=x_mu, scale=x_sigma)

    vars, var_approximations, data =
        model_nondisjoint_feature_expression(
            input, num_features, feature_idxs, feature_transcript_idxs,
            antifeature_idxs, antifeature_transcript_idxs, x)


    # inference
    # ---------

    qx_loc = tf.Variable(tf.zeros([num_samples, num_features]), name="qx_loc")
    qx_softplus_scale = tf.Variable(tf.fill([num_samples, num_features], -2.0), name="qx_softplus_scale")
    qx = edmodels.NormalWithSoftplusScale(loc=qx_loc, scale=qx_softplus_scale,
                                          name="qx")

    qmu_bias_loc = tf.Variable(tf.zeros([1, num_features]), name="qmu_bias_loc")
    qmu_bias_softplus_scale = tf.Variable(tf.fill([1, num_features], -1.0f0), name="qmu_bias_softplus_scale")
    qmu_bias = edmodels.NormalWithSoftplusScale(loc=qmu_bias_loc,
                                                scale=qmu_bias_softplus_scale,
                                                name="qmu_bias")

    qx_sigma_sq_mu_param    = tf.Variable(tf.fill([num_features], -3.0f0), name="qx_sigma_sq_mu_param")
    qx_sigma_sq_sigma_param = tf.Variable(tf.fill([num_features], -4.0f0), name="qx_sigma_sq_sigma_param")
    qx_sigma_sq = edmodels.TransformedDistribution(
        distribution=edmodels.NormalWithSoftplusScale(qx_sigma_sq_mu_param, qx_sigma_sq_sigma_param),
        bijector=tfdist.bijectors[:Exp](),
        name="qx_sigma_sq")

    qw_loc = tf.Variable(tf.random_normal([num_features, num_components]), name="qw_loc")
    qw_softplus_scale = tf.Variable(tf.fill([num_features, num_components], -2.0f0), name="qw_softplus_scale")
    qw = edmodels.NormalWithSoftplusScale(loc=qw_loc, scale=qw_softplus_scale, name="qw")

    qz_loc = tf.Variable(tf.random_normal([num_samples, num_components]), name="qz_loc")
    qz_softplus_scale = tf.Variable(tf.fill([num_samples, num_components], -2.0f0), name="qz_softplus_scale")
    qz = edmodels.NormalWithSoftplusScale(loc=qz_loc, scale=qz_softplus_scale, name="qz")

    var_approximations[x] = qx
    var_approximations[mu_bias] = qmu_bias
    var_approximations[x_sigma_sq] = qx_sigma_sq
    var_approximations[z] = qz
    var_approximations[w] = qw

    inference = ed.KLqp(latent_vars=PyDict(var_approximations),
                        data=PyDict(data))

    # optimizer = tf.train[:AdamOptimizer](0.05)
    optimizer = tf.train[:AdamOptimizer](0.1)
    run_inference(input, inference, 10000, optimizer)

    sess = ed.get_session()
    qz_loc_values = sess[:run](qz_loc)

    open("splicing-pca-estimates.csv", "w") do out
        print(out, "sample,")
        println(out, join([string("pc", j) for j in 1:num_components], ','))
        for i in 1:num_samples
            print(out, '"', input.loaded_samples.sample_names[i], '"', ',')
            println(out, join(qz_loc_values[i,:], ','))
        end
    end

    # qw_loc_values = sess[:run](qw_loc)
    # open("splicing-pca-coefficients.csv", "w") do out
    #     println(out, "component,id,transcript_id,coeff")
    #     for t in input.ts
    #         i = t.metadata.id
    #         for j in 1:num_components
    #             @printf(out, "%d,%d,%s,%e\n", j, i, t.metadata.name, qw_loc_values[i, j])
    #         end
    #     end
    # end
end


EXTRUDER_MODELS["pca"] = estimate_pca
EXTRUDER_MODELS["batch-pca"] = input -> estimate_pca(input, correct_batch_effects=true)

