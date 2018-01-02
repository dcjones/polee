
function estimate_linear_regression(input::ModelInput)
    if input.feature == :transcript
        estimate_transcript_linear_regression(input)
    elseif input.feature == :splicing
        estimate_splicing_linear_regression(input)
    else
        error("Linear regression estimates for $(input.feature) not supported")
    end
end


function build_linear_regression_design_matrix(input::ModelInput)
    num_samples, n = size(input.loaded_samples.x0_values)
    factoridx = Dict{String, Int}()
    factoridx["bias"] = 1
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
    X_[:, factoridx["bias"]] = 1
    X = tf.constant(X_)

    return num_factors, factoridx, X
end


function estimate_transcript_linear_regression(input::ModelInput)

    num_samples, n = size(input.loaded_samples.x0_vaues)
    num_factors, factoridx, X = build_linear_regression_design_matrix(input)

    println("Sample data loaded")

    # model specification
    # -------------------

    w_mu0 = 0.0
    w_sigma0 = 1.0
    w_bias_mu0 = log(1/n)
    w_bias_sigma0 = 5.0

    w_sigma = tf.concat(
                  [tf.constant(w_bias_sigma0, shape=[1, n]),
                   tf.constant(w_sigma0, shape=[num_factors-1, n])], 0)
    w_mu = tf.concat(
                  [tf.constant(w_bias_mu0, shape=[1, n]),
                   tf.constant(w_mu0, shape=[num_factors-1, n])], 0)

    w = edmodels.MultivariateNormalDiag(name="W", w_mu, w_sigma)

    x = tf.matmul(X, w)

    likapprox_laparam = rnaseq_approx_likelihood.RNASeqApproxLikelihood(
                    x=x,
                    efflens=input.likapprox_efflen,
                    invhsb_params=input.likapprox_invhsb_params,
                    value=input.likapprox_laparam)

    # inference
    # ---------

    println("Estimating...")

    qw_loc = tf.Variable(tf.multiply(0.001, tf.random_normal([num_factors, n])))
    qw_scale = tf.nn[:softplus](tf.Variable(tf.zeros([num_factors, n])))
    qw = edmodels.Normal(loc=qw_loc, scale=qw_scale)

    inference = ed.KLqp(Dict(w => qw),
                        data=Dict(likapprox_laparam => input.likapprox_laparam))

    optimizer = tf.train[:AdamOptimizer](0.1)
    inference[:run](n_iter=1000, optimizer=optimizer)


    sess = ed.get_session()

    output_filename = isnull(input.output_filename) ?
        "effects.db" : get(input.output_filename)

    @time write_effects(output_filename, factoridx,
                        sess[:run](qw_loc),
                        sess[:run](qw_scale),
                        input.feature)
end


function estimate_splicing_linear_regression(input::ModelInput)
    num_samples, n = size(input.loaded_samples.x0_values)
    num_factors, factoridx, X = build_linear_regression_design_matrix(input)

    (num_features,
     feature_idxs, feature_transcript_idxs,
     antifeature_idxs, antifeature_transcript_idxs) = splicing_features(input)

    # Model relative feature expression with linear regression

    w_mu0 = 0.0
    w_sigma0 = 1.0
    w_bias_mu0 = log(1/num_features)
    w_bias_sigma0 = 5.0

    w_sigma = tf.concat(
                  [tf.constant(w_bias_sigma0, shape=[1, num_features]),
                   tf.constant(w_sigma0, shape=[num_factors-1, num_features])], 0)
    w_mu = tf.concat(
                  [tf.constant(w_bias_mu0, shape=[1, num_features]),
                   tf.constant(w_mu0, shape=[num_factors-1, num_features])], 0)

    w = edmodels.MultivariateNormalDiag(name="W", w_mu, w_sigma)

    x_feature = tf.matmul(X, w)


    vars, var_approximations, data =
        model_nondisjoint_feature_expression(input, num_features,
                                            feature_idxs, feature_transcript_idxs,
                                            antifeature_idxs, antifeature_transcript_idxs,
                                            x_feature)

    qw_loc = tf.Variable(tf.multiply(0.001, tf.random_normal([num_factors, num_features])))
    qw_scale = tf.nn[:softplus](tf.Variable(tf.zeros([num_factors, num_features])))
    qw = edmodels.Normal(loc=qw_loc, scale=qw_scale)

    vars[:w] = w
    var_approximations[w] = qw

    inference = ed.KLqp(latent_vars=var_approximations, data=data)

    optimizer = tf.train[:AdamOptimizer](5e-2)
    run_inference(input, inference, 2000, optimizer)

    sess = ed.get_session()

    output_filename = isnull(input.output_filename) ?
        "effects.db" : get(input.output_filename)

    # TODO: need a way to output the features and such
    @time write_effects(output_filename, factoridx,
                        sess[:run](qw_loc),
                        sess[:run](qw_scale),
                        input.feature)
end


EXTRUDER_MODELS["linear-regression"] = estimate_linear_regression


