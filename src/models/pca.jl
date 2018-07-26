

function estimate_pca(input::ModelInput; num_components::Int=2,
                      correct_batch_effects::Bool=false)

    if input.feature == :transcript
        estimate_transcript_pca(input, num_components=num_components,
                                correct_batch_effects=correct_batch_effects)
    elseif input.feature == :gene
        estimate_gene_pca(input, num_components=num_components,
                          correct_batch_effects=correct_batch_effects)
    elseif input.feature == :splicing
        estimate_splicing_pca(input, num_components=num_components,
                              correct_batch_effects=correct_batch_effects)
    else
        error("$(input.feature) feature not supported by pca")
    end
end



"""
Simple probabalistic PCA with fixed variance. Useful for finding a good
initialization to the full model.
"""
function simple_pca(input, x_obs ; num_components::Int=2)
    num_samples, num_features = size(x_obs)

    mu_bias_mu0 = log(1f0/num_features)
    mu_bias_sigma0 = 4.0f0

    mu_bias = edmodels.Normal(loc=tf.fill([num_features], mu_bias_mu0),
                              scale=tf.fill([num_features], mu_bias_sigma0))

    w = edmodels.Normal(loc=tf.zeros([num_features, num_components]),
                        scale=tf.fill([num_features, num_components], 1.0f0))
    z = edmodels.Normal(loc=tf.zeros([num_samples, num_components]),
                        scale=tf.fill([num_samples, num_components], 1.0f0))

    mu_pca = tf.matmul(z, w, transpose_b=true)

    x_mu = tf.add(mu_bias, mu_pca)

    x_err_sigma = 0.1f0
    x = edmodels.Normal(
        loc=x_mu, scale=tf.fill([num_samples, num_features], 0.1))

    mu_bias_obs = mean(x_obs, 1)[1,:]
    vars = Dict(
        w       => edmodels.PointMass(tf.Variable(
            tf.multiply(0.001f0, tf.random_normal([num_features, num_components])))),
        z       => edmodels.PointMass(tf.Variable(
            tf.multiply(0.001f0, tf.random_normal([num_samples, num_components])))),
        mu_bias => edmodels.PointMass(tf.Variable(mu_bias_obs)))

    inference = ed.MAP(vars, data=Dict(x => x_obs))
    optimizer = tf.train[:AdamOptimizer](2e-2)
    run_inference(input, inference, 1000, optimizer)

    sess = ed.get_session()
    return (sess[:run](vars[mu_bias]), sess[:run](vars[z]), sess[:run](vars[w]))
end



function estimate_transcript_pca(input::ModelInput; num_components::Int=2,
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

    x_sigma_alpha0 = tf.constant(SIGMA_ALPHA0, shape=[n])
    x_sigma_beta0 = tf.constant(SIGMA_BETA0, shape=[n])
    x_sigma_sq = edmodels.InverseGamma(x_sigma_alpha0, x_sigma_beta0)

    # x = edmodels.StudentT(df=10.0f0, loc=x_mu, scale=x_sigma)
    x = edmodels.Normal(loc=x_mu, scale=tf.sqrt(x_sigma_sq))

    likapprox = RNASeqApproxLikelihood(input, x)

    x_init = log.(input.loaded_samples.x0_values)
    qmu_init, qz_init, qw_init = simple_pca(input, x_init)

    # inference
    # ---------

    if input.inference == :map || input.inference == :default
        vars = Dict(
            mu_bias =>
                edmodels.PointMass(tf.Variable(qmu_init, name="qmu_bias_param")),
            x_sigma_sq =>
                edmodels.PointMass(tf.nn[:softplus](tf.Variable(
                    tf.fill([n], -2.0f0), name="qx_sigma_q_param"))),
            x =>
                edmodels.PointMass(tf.Variable(qmu_init, name="qmu_param")),
            w =>
                edmodels.PointMass(tf.Variable(qw_init, name="qw_param")),
            z =>
                edmodels.PointMass(tf.Variable(qz_init, name="qz_param")))

        if correct_batch_effects
            vars[w_batch] = edmodels.PointMass(
                tf.Variable(fill(0.0f0, (num_factors, n)), name="qw_batch_parlam") )
        end

        inference = ed.MAP(vars, data=Dict(likapprox => Float32[]))
        optimizer = tf.train[:AdamOptimizer](5e-2)
        run_inference(input, inference, 1000, optimizer)

        sess = ed.get_session()
        qz_loc_values = sess[:run](qz)
        qw_loc_values = sess[:run](qw)
    else
        error("Inference method $(input.inference) not supported for PCA.")
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


function estimate_gene_pca(input::ModelInput; num_components::Int=2,
                               correct_batch_effects::Bool=false)

    if input.inference == :default
        input.inference = :map
    end

    num_samples, n = size(input.loaded_samples.x0_values)
    num_features, gene_idxs, transcript_idxs, gene_ids, gene_names =
        gene_feature_matrix(input.ts, input.ts_metadata)
    num_aux_features = regularize_disjoint_feature_matrix!(gene_idxs, transcript_idxs, n)
    num_features += num_aux_features

    # bias
    # ----

    mu_bias_mu0 = log(1.0f0/num_features)
    mu_bias_sigma0 = 4.0f0

    mu_bias = edmodels.Normal(
        loc=tf.fill([num_features], mu_bias_mu0),
        scale=tf.fill([num_features], mu_bias_sigma0))

    # pca
    # ---

    w = edmodels.Normal(
        loc=tf.zeros([num_features, num_components]),
        scale=tf.fill([num_features, num_components], 1.0f0))
    z = edmodels.Normal(
        loc=tf.zeros([num_samples, num_components]),
        scale=tf.fill([num_samples, num_components], 1.0f0))

    mu_pca = tf.matmul(z, w, transpose_b=true)
    x_mu = tf.add(mu_bias, mu_pca)

    x_sigma_sq_alpha0 = tf.constant(SIGMA_ALPHA0, shape=[num_features])
    x_sigma_sq_beta0 = tf.constant(SIGMA_BETA0, shape=[num_features])
    x_sigma_sq = edmodels.InverseGamma(x_sigma_sq_alpha0, x_sigma_sq_beta0)

    x = edmodels.Normal(loc=x_mu, scale=tf.sqrt(x_sigma_sq))

    vars, var_approximations, data =
        model_disjoint_feature_expression(input, gene_idxs, transcript_idxs,
                                          num_features, x)

    # inference
    # ---------

    # find good initial values
    qx_init = zeros(Float32, (num_samples, num_features))
    for i in 1:num_samples
        for (j, k) in zip(gene_idxs, transcript_idxs)
            qx_init[i, j] += input.loaded_samples.x0_values[i, k]
        end
    end
    map!(log, qx_init, qx_init)
    qmu_init, qz_init, qw_init = simple_pca(
        input, qx_init; num_components=num_components)

    if input.inference == :map
        var_approximations = merge(var_approximations, Dict(
            x =>
                edmodels.PointMass(tf.Variable(qx_init, name="qx_param")),
            x_sigma_sq =>
                edmodels.PointMass(tf.nn[:softplus](tf.Variable(
                tf.fill([num_features], -2.0f0), name="qx_sigma_sq_param"))),
            mu_bias =>
                edmodels.PointMass(tf.Variable(qmu_init, name="qw_bias_param")),
            w =>
                edmodels.PointMass(tf.Variable(qw_init, name="qw_param")),
            z =>
                edmodels.PointMass(tf.Variable(qz_init, name="qz_param"))))

        inference = ed.MAP(var_approximations, data=data)
        optimizer = tf.train[:AdamOptimizer](5e-2)
        run_inference(input, inference, 1000, optimizer)

        sess = ed.get_session()
        qz_loc_values = sess[:run](var_approximations[z])
        qw_loc_values = sess[:run](var_approximations[w])
    else
        error("Inference type $(input.inference) not supported for gene-level PCA")
    end

    open("gene-pca-estimates.csv", "w") do out
        print(out, "sample,")
        println(out, join([string("pc", j) for j in 1:num_components], ','))
        for i in 1:num_samples
            print(out, '"', input.loaded_samples.sample_names[i], '"', ',')
            println(out, join(qz_loc_values[i,:], ','))
        end
    end
end


function estimate_splicing_pca(input::ModelInput; num_components::Int=2,
                               correct_batch_effects::Bool=false)

    if input.inference == :default
        input.inference = :map
    end

    splice_loc_param, splice_scale_param = approximate_splicing_likelihood(input)

    num_samples, n = size(input.loaded_samples.x0_values)
    num_features = size(splice_loc_param, 2)

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

    mu_bias = edmodels.Normal(
        loc=tf.fill([num_features], w_bias_mu0), scale=w_bias_sigma0,
        name="mu_bias")

    # pca model specification
    # -----------------------

    w = edmodels.Normal(
        loc=tf.zeros([num_features, num_components]), scale=1.0f0,
        name="w")

    z = edmodels.Normal(
        loc=tf.zeros([num_samples, num_components]), scale=1.0f0,
        name="z")

    mu_pca = tf.matmul(z, w, transpose_b=true)
    x_mu = tf.add(mu_bias, mu_pca)

    # TODO:
    # if correct_batch_effects
    #     x_mu = tf.add(mu_batch, x_mu)
    # end

    x_sigma_sq_alpha0 = tf.constant(SIGMA_ALPHA0, shape=[num_features])
    x_sigma_sq_beta0 = tf.constant(SIGMA_BETA0, shape=[num_features])
    x_sigma_sq = edmodels.InverseGamma(
        x_sigma_sq_alpha0, x_sigma_sq_beta0, name="x_err_sigma")

    x = edmodels.Normal(loc=x_mu, scale=tf.sqrt(x_sigma_sq), name="x")

    approxlik = polee_py.ApproximatedLikelihood(
        edmodels.NormalWithSoftplusScale(
            loc=splice_loc_param, scale=splice_scale_param),
        x)

    data = Dict(approxlik => zeros(Float32, (num_samples, 0)))

    # inference
    # ---------

    qx_init = splice_loc_param
    qmu_init, qz_init, qw_init = simple_pca(input, qx_init, num_components=num_components)

    if input.inference == :map
        qz_param = tf.Variable(tf.random_normal([num_samples, num_components]), name="qz")

        var_approximations = Dict(
            x =>
                edmodels.PointMass(
                    tf.Variable(qx_init, name="qx_param")),
            mu_bias =>
                edmodels.PointMass(
                    tf.Variable(qmu_init, name="qmu_bias_param")),
            x_sigma_sq =>
                edmodels.PointMass(tf.nn[:softplus](
                    tf.Variable(
                        tf.fill([num_features], -2.0f0),
                        name="qx_sigma_sq_param"))),
            w =>
                edmodels.PointMass(
                    tf.Variable(qw_init, name="qw_param")),
            z =>
                edmodels.PointMass(
                    tf.Variable(qz_init, name="qz_param")))

        inference = ed.MAP(var_approximations, data=data)
        optimizer = tf.train[:AdamOptimizer](2e-2)
        run_inference(input, inference, 5000, optimizer)

        sess = ed.get_session()
        z_est = sess[:run](var_approximations[z])

    else
        error("Inference type $(input.inference) is not supported by splicing PCA.")
    end

    output_filename = isnull(input.output_filename) ?
        "splicing-pca-estimates.csv" : get(input.output_filename)

    open(output_filename, "w") do out
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


POLEE_MODELS["pca"] = estimate_pca
POLEE_MODELS["batch-pca"] = input -> estimate_pca(input, correct_batch_effects=true)

