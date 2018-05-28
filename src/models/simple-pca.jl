
function estimate_simple_pca(input::ModelInput; num_components::Int=2,
                      correct_batch_effects::Bool=false)

    if input.feature == :transcript
        estimate_simple_transcript_pca(input, num_components=num_components,
                                       correct_batch_effects=correct_batch_effects)
    elseif input.feature == :gene
        error()
    elseif input.feature == :splicing
        estimate_simple_splicing_pca(input, num_components=num_components)
    else
        error("$(input.feature) feature not supported by simple-pca")
    end
end


function estimate_simple_transcript_pca(input::ModelInput; num_components::Int=2,
                                        correct_batch_effects::Bool=false)

    num_samples, n = size(input.loaded_samples.x0_values)
    x_obs, _ = estimate_simple_transcript_expression(input, false)

    # upper quantile normalization
    # ----------------------------

    qs = Array{Float32}(num_samples)
    x_obs = exp.(x_obs)
    for i in 1:num_samples
        qs[i] = quantile(x_obs[i,:], 0.9)
    end
    qs ./= mean(qs)

    for i in 1:num_samples
        for j in 1:n
            x_obs[i,j] /= qs[i]
        end
    end
    x_obs = log.(x_obs)

    @show typeof(x_obs)
    @show size(x_obs)


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

    mu_bias = edmodels.Normal(loc=tf.fill([n], w_bias_mu0),
                              scale=tf.fill([n], w_bias_sigma0))

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

    # x_sigma_alpha0 = tf.constant(SIGMA_ALPHA0, shape=[n])
    # x_sigma_beta0 = tf.constant(SIGMA_BETA0, shape=[n])
    # x_sigma = edmodels.InverseGamma(x_sigma_alpha0, x_sigma_beta0)
    x_sigma = 0.1f0
    # x = edmodels.StudentT(df=10.0f0, loc=x_mu, scale=x_sigma)
    x = edmodels.Normal(loc=x_mu, scale=x_sigma)

    # likapprox = RNASeqApproxLikelihood(input, x_mu)

    # inference
    # ---------

    x0_log = tf.log(tf.constant(input.loaded_samples.x0_values))
    datadict = PyDict(Dict(x => x_obs))

    if input.inference == :variational
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

        qx_sigma_mu_param    = tf.Variable(tf.fill([n], -2.0f0), name="qx_sigma_mu_param")
        qx_sigma_sigma_param = tf.Variable(tf.fill([n], -1.0f0), name="qx_sigma_sigma_param")
        qx_sigma = edmodels.TransformedDistribution(
            distribution=edmodels.NormalWithSoftplusScale(qx_sigma_mu_param, qx_sigma_sigma_param),
            bijector=tfdist.bijectors[:Exp](),
            name="LogNormalTransformedDistribution")

        # vars = Dict(x => qx, w => qw, z => qz, mu_bias => qmu_bias,
        #             # x_sigma => qx_sigma)
        vars = Dict(w => qw, z => qz, mu_bias => qmu_bias,
                    x_sigma => qx_sigma)

        if correct_batch_effects
            qw_batch_loc = tf.Variable(fill(0.0f0, (num_factors, n)), name="qw_batch_loc")
            qw_batch_softplus_scale = tf.Variable(tf.fill([num_factors, n], -1.0f0), name="qw_batch_softplus_scale")
            qw_batch = edmodels.NormalWithSoftplusScale(loc=qw_batch_loc, scale=qw_batch_softplus_scale)

            vars[w_batch] = qw_batch
        end

        inference = ed.KLqp(vars, data=datadict)

        optimizer = tf.train[:AdamOptimizer](0.05)
        run_inference(input, inference, 1000, optimizer)

        sess = ed.get_session()
        qz_loc_values = sess[:run](qz_loc)
        qw_loc_values = sess[:run](qw_loc)
    elseif input.inference == :map
        qx_loc = tf.Variable(x0_log, name="qx_loc")
        qx = edmodels.PointMass(qx_loc)

        qmu_bias_loc = tf.Variable(tf.reduce_mean(x0_log, 0), name="qmu_bias_loc")
        qmu_bias = edmodels.PointMass(qmu_bias_loc)

        qw_loc = tf.Variable(tf.multiply(0.001f0, tf.random_normal([n, num_components])), name="qw_loc")
        qw = edmodels.PointMass(qw_loc)

        qz_loc = tf.Variable(tf.multiply(0.001f0, tf.random_normal([num_samples, num_components])), name="qz_loc")
        qz = edmodels.PointMass(qz_loc)

        # qx_sigma_loc = tf.Variable(tf.fill([n], -2.0f0), name="qx_sigma_loc_param")
        # qx_sigma = edmodels.PointMass(tf.nn[:softplus](qx_sigma_loc))
        # qx_sigma_mu_param    = tf.Variable(tf.fill([n], -2.0f0), name="qx_sigma_mu_param")
        # qx_sigma_sigma_param = tf.Variable(tf.fill([n], -1.0f0), name="qx_sigma_sigma_param")
        # qx_sigma = edmodels.TransformedDistribution(
        #     distribution=edmodels.NormalWithSoftplusScale(qx_sigma_mu_param, qx_sigma_sigma_param),
        #     bijector=tfdist.bijectors[:Exp](),
        #     name="LogNormalTransformedDistribution")

        vars = Dict(x => qx, w => qw, z => qz, mu_bias => qmu_bias)
        # vars = Dict(w => qw, z => qz, mu_bias => qmu_bias)

        if correct_batch_effects
            qw_batch_loc = tf.Variable(fill(0.0f0, (num_factors, n)), name="qw_batch_loc")
            qw_batch = edmodels.PointMass(qw_batch_loc)

            vars[w_batch] = qw_batch
        end

        @show vars
        @show x
        inference = ed.MAP(vars, data=datadict)

        optimizer = tf.train[:AdamOptimizer](0.05)
        run_inference(input, inference, 2000, optimizer)

        sess = ed.get_session()
        qz_loc_values = sess[:run](qz)
        qw_loc_values = sess[:run](qw)
    end

    output_filename = isnull(input.output_filename) ?
        "transcript-pca-estimates.csv" : get(input.output_filename)

    open(output_filename, "w") do out
        print(out, "sample,")
        println(out, join([string("pc", j) for j in 1:num_components], ','))
        for i in 1:num_samples
            print(out, '"', input.loaded_samples.sample_names[i], '"', ',')
            println(out, join(qz_loc_values[i,:], ','))
        end
    end

    open("transcript-pca-coefficients.csv", "w") do out
        println(out, "component,id,transcript_id,coeff")
        for t in input.ts
            i = t.metadata.id
            for j in 1:num_components
                @printf(out, "%d,%d,%s,%e\n", j, i, t.metadata.name, qw_loc_values[i, j])
            end
        end
    end

end


function estimate_simple_splicing_pca(input::ModelInput; num_components::Int=2)

    x_obs = estimate_splicing_proportions(input, write_results=false)

    num_samples, n = size(input.loaded_samples.x0_values)
    num_features = size(x_obs, 2)

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
                              scale=tf.fill([1, num_features], w_bias_sigma0),
                              name="mu_bias")

    # pca model specification
    # -----------------------

    w = edmodels.Normal(loc=tf.zeros([num_components, num_features]),
                        scale=tf.fill([num_components, num_features], 1.0f0))

    z = edmodels.Normal(loc=tf.zeros([num_samples, num_components]),
                        scale=tf.fill([num_samples, num_components], 1.0f0))

    mu_pca = tf.matmul(z, w)
    # mu_pca = tf_print_span(mu_pca, "mu_pca span")

    x_mu = tf.add(mu_bias, mu_pca)
    # x_mu = mu_bias

    # TODO:
    # if correct_batch_effects
    #     x_mu = tf.add(mu_batch, x_mu)
    # end

    x_err_sigma_alpha0 = tf.constant(SIGMA_ALPHA0, shape=[1, num_features])
    x_err_sigma_beta0 = tf.constant(SIGMA_BETA0, shape=[1, num_features])
    x_err_sigma = edmodels.InverseGamma(x_err_sigma_alpha0, x_err_sigma_beta0, name="x_err_sigma")
    # x_err_sigma = 0.1

    # x_err = edmodels.Normal(loc=tf.fill([num_samples, num_features], 0.0f0), scale=x_err_sigma, name="x_err")
    x = edmodels.Normal(loc=x_mu, scale=x_err_sigma, name="x_err")

    # x_feature = edmodels.StudentT(df=10.0, loc=x_mu, scale=x_feature_sigma)
    # x_feature_ = tf.Print(x_feature, [x_feature])
    # x_feature_ = tf_print_span(x_feature, "x_feature")

    datadict = Dict(x => x_obs)

    # inference
    # ---------

    if input.inference == :variational
        qx_err_loc = tf.Variable(tf.zeros([num_samples, num_features]), name="qx_err_loc")
        qx_err_softplus_scale = tf.Variable(tf.fill([num_samples, num_features], -2.0f0), name="qx_err_softplus_scale")
        qx_err = edmodels.NormalWithSoftplusScale(loc=qx_err_loc, scale=qx_err_softplus_scale, name="qx_err")

        qmu_bias_loc = tf.Variable(tf.reduce_mean(splice_loc_param, 0), name="qmu_bias_loc")
        qmu_bias_softplus_scale = tf.Variable(tf.fill([1, num_features], -1.0f0), name="qmu_bias_softplus_scale")
        qmu_bias = edmodels.NormalWithSoftplusScale(loc=qmu_bias_loc,
                                                    scale=qmu_bias_softplus_scale,
                                                    name="qmu_bias")

        qx_err_sigma_mu_param =
            tf.Variable(tf.fill([1, num_features], -3.0f0), name="qx_sigma_mu_param")
        # qx_err_sigma_sq_mu_param = tf_print_span(qx_err_sigma_sq_mu_param, "qx_err_sigma_sq_mu_param")
        qx_err_sigma_sigma_param =
            tf.Variable(tf.fill([1, num_features], -1.0f0), name="qx_sigma_sigma_param")
        # qx_err_sigma_sq_sigma_param = tf_print_span(qx_err_sigma_sq_sigma_param, "qx_err_sigma_sq_sigma_param")
        qx_err_sigma = edmodels.TransformedDistribution(
            distribution=edmodels.NormalWithSoftplusScale(
                qx_err_sigma_mu_param, qx_err_sigma_sigma_param),
            bijector=tfdist.bijectors[:Softplus](),
            name="qx_err_sigma_sq")

        qw_loc = tf.Variable(0.001f0 * tf.random_normal([num_components, num_features]), name="qw_loc")
        # qw_loc = tf_print_span(qw_loc, "qw_loc")
        qw_softplus_scale = tf.Variable(tf.fill([num_components, num_features], -1.0f0), name="qw_softplus_scale")
        qw = edmodels.NormalWithSoftplusScale(loc=qw_loc, scale=qw_softplus_scale, name="qw")

        qz_loc = tf.Variable(0.001f0 * tf.random_normal([num_samples, num_components]), name="qz_loc")
        # qz_loc = tf_print_span(qz_loc, "qz_loc")
        qz_softplus_scale = tf.Variable(tf.fill([num_samples, num_components], -8.0f0), name="qz_softplus_scale")
        qz = edmodels.NormalWithSoftplusScale(loc=qz_loc, scale=qz_softplus_scale, name="qz")

        var_approximations = Dict()
        var_approximations[x_err] = qx_err
        var_approximations[mu_bias] = qmu_bias
        var_approximations[x_err_sigma] = qx_err_sigma
        var_approximations[z] = qz
        var_approximations[w] = qw

        inference = ed.KLqp(var_approximations, data=data)

        # optimizer = tf.train[:AdamOptimizer](0.001)
        optimizer = tf.train[:AdamOptimizer](0.05)
        run_inference(input, inference, 2000, optimizer)

        sess = ed.get_session()
        z_est = sess[:run](qz_loc)

        # scl = tf.sqrt(qx_feature_sigma_sq[:value]())
        # scl = tf.sqrt(x_feature_sigma_sq[:value]())

        # scl = sess[:run](qx_feature_sigma_sq[:value]())
        # @show extrema(scl)

        # dist = edmodels.Normal(loc=tf.fill([num_samples, num_features], 0.0f0), scale=scl)

        # smpl = sess[:run](qx_feature[:value]())
        # @show extrema(smpl)
        # lps = sess[:run](dist[:log_prob](smpl))[1,:]
        # p = sortperm(lps)
        # # @show p[1:10]
        # @show lps[p][1:10]
        # @show scl[p][1:10]
        # @show smpl[p][1:10]
        # @show sess[:run](qx_feature_loc)[p][1:10]
        # @show sess[:run](tf.nn[:softplus](qx_feature_softplus_scale))[p][1:10]

    elseif input.inference == :map

        qz_param = tf.Variable(tf.random_normal([num_samples, num_components]), name="qz")

        var_maps = Dict(
            x => edmodels.PointMass(
                tf.Variable(tf.zeros([num_samples, num_features]), name="qx")),
            mu_bias   => edmodels.PointMass(
                tf.Variable(tf.zeros([1, num_features]), name="qmu_bias")),
            # x_err_sigma => edmodels.PointMass(
            #     tf.nn[:softplus](tf.Variable(tf.fill([1, num_features], 0.0f0), name="qx_feature_sigma"))),
            w => edmodels.PointMass(
                tf.Variable(tf.random_normal([num_components, num_features]), name="qw")),
            z => edmodels.PointMass(qz_param),
        )

        inference = ed.MAP(var_maps, data=datadict)
        optimizer = tf.train[:AdamOptimizer](2e-2)
        run_inference(input, inference, 5000, optimizer)

        sess = ed.get_session()
        z_est = sess[:run](qz_param)

    else
        error("Inference type $(input.inference) is not supported.")
    end

    open("splicing-pca-estimates.csv", "w") do out
        print(out, "sample,")
        println(out, join([string("pc", j) for j in 1:num_components], ','))
        for i in 1:num_samples
            print(out, '"', input.loaded_samples.sample_names[i], '"', ',')
            println(out, join(z_est[i,:], ','))
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


POLEE_MODELS["simple-pca"] = estimate_simple_pca