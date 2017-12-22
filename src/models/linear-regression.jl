
function estimate_linear_regression(input::ModelInput)
    if input.feature != :transcript
        error("Linear regression only implemented with transcripts")
    end

    num_samples, n = size(input.x0)

    # build design matrix
    # -------------------

    factoridx = Dict{String, Int}()
    factoridx["bias"] = 1
    for factors in input.sample_factors
        for factor in factors
            get!(factoridx, factor, length(factoridx) + 1)
        end
    end

    num_factors = length(factoridx)
    X_ = zeros(Float32, (num_samples, num_factors))
    for i in 1:num_samples
        for factor in input.sample_factors[i]
            j = factoridx[factor]
            X_[i, j] = 1
        end
    end
    X_[:, factoridx["bias"]] = 1
    X = tf.constant(X_)

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


EXTRUDER_MODELS["linear-regression"] = estimate_linear_regression


