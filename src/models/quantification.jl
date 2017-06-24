


function estimate_expression(input::ModelInput)
    if input.feature == :transcript
        return estimate_transcript_expression(input)
    elseif input.feature == :gene
        return estimate_gene_expression(input)
    elseif input.feature == :splicing
        return estimate_splicing_expression(input)
    else
        error("Expression estimates for $(feature)s not supported")
    end
end


function estimate_transcript_expression(input::ModelInput)
    qy_mu_value, qy_sigma_value =
        estimate_expression(input.likapprox_data, input.y0, input.sample_factors)
end


function estimate_gene_expression(input::ModelInput)
    m, I, J, names = gene_feature_matrix(input.ts, input.ts_metadata)
    F = tf.SparseTensor(indices=cat(2, I-1, J-1), values=tf.ones(length(I)),
                        dense_shape=[m, length(input.ts)])

    qy_mu_value, qy_sigma_value =
        estimate_feature_expression(input.likapprox_data, input.y0, input.sample_factors, F)
end


function estimate_splicing_expression(input::ModelInput)
    cassette_exons = get_cassette_exons(input.ts)

    # TODO: also include alt acceptor/donor sites and maybe alt UTRs

    I = Int[]
    J = Int[]
    for (i, (intron, flanks)) in  enumerate(cassette_exons)
        for id in intron.metadata
            push!(I, 2*i-1)
            push!(J, id)
        end

        for id in flanks.metadata[3]
            push!(I, 2*i)
            push!(J, id)
        end
    end
    m = 2*length(cassette_exons)

    F = tf.SparseTensor(indices=cat(2, I-1, J-1), values=tf.ones(length(I)),
                        dense_shape=[m, length(input.ts)])

    qy_mu_value, qy_sigma_value =
        estimate_feature_expression(input.likapprox_data, input.y0, input.sample_factors, F)
end


function transcript_quantification_model(likapprox_data, y0)
    num_samples, n = y0[:get_shape]()[:as_list]()

    # y_mu: pooled mean
    y_mu_mu0 = tf.constant(log(1/n), shape=[n])
    y_mu_sigma0 = tf.constant(10.0, shape=[n])
    y_mu = edmodels.MultivariateNormalDiag(y_mu_mu0, y_mu_sigma0)

    # y_sigma: variance around pooled mean
    y_sigma_mu0 = tf.constant(0.0, shape=[n])
    y_sigma_sigma0 = tf.constant(1.0, shape=[n])
    y_log_sigma = edmodels.MultivariateNormalDiag(y_sigma_mu0, y_sigma_sigma0)
    y_sigma = tf.exp(y_log_sigma)

    # y: quantification
    y_mu_param = tf.matmul(tf.ones([num_samples, 1]),
                           tf.expand_dims(y_mu, 0))

    y_sigma_param = tf.matmul(tf.ones([num_samples, 1]),
                              tf.expand_dims(y_sigma, 0))

    y = edmodels.MultivariateNormalDiag(y_mu_param, y_sigma_param)

    likapprox = rnaseq_approx_likelihood.RNASeqApproxLikelihood(
                  y=y, value=likapprox_data)

    return y, y_mu_param, y_sigma_param, y_mu, likapprox
end


function estimate_feature_expression(likapprox_data, y0, sample_factors, features)
    num_samples, n = y0[:get_shape]()[:as_list]()
    num_features = features[:get_shape]()[:as_list]()[1]
    y, y_mu_param, y_sigma_param, y_mu, likapprox =
        transcript_quantification_model(likapprox_data, y0)

    y_features_mu_param =
        tf.transpose(tf.sparse_tensor_dense_matmul(features, y_mu_param, adjoint_b=true))
    y_features_sigma_param =
        tf.transpose(tf.sparse_tensor_dense_matmul(features, y_sigma_param, adjoint_b=true))
    y_features = edmodels.MultivariateNormalDiag(y_features_mu_param, y_features_sigma_param)

    println("Estimating...")

    # y approximation
    qy_mu_param = tf.Variable(y0)
    qy_sigma_param = tf.identity(tf.Variable(tf.fill([num_samples, n], 0.1)))
    qy = edmodels.MultivariateNormalDiag(qy_mu_param, qy_sigma_param)

    # y_feature approximation
    qy_features_mu_param = tf.Variable(
        tf.transpose(tf.sparse_tensor_dense_matmul(features, y0, adjoint_b=true)))
    qy_features_sigma_param =
        tf.identity(tf.Variable(tf.fill([num_samples, num_features], 0.1)))
    qy_features = edmodels.MultivariateNormalDiag(qy_features_mu_param,
                                                  qy_features_sigma_param)
    # y_mu approximation
    qy_mu_mu_param = tf.Variable(y0[1])
    qy_mu_sigma_param = tf.identity(tf.Variable(tf.fill([n], 0.1)))
    qy_mu = edmodels.MultivariateNormalDiag(qy_mu_mu_param, qy_mu_sigma_param)

    inference = ed.KLqp(PyDict(Dict(y => qy, y_features => qy_features, y_mu => qy_mu)),
                        data=PyDict(Dict(likapprox => likapprox_data)))

    optimizer = tf.train[:AdamOptimizer](1e-2)
    inference[:run](n_iter=250, optimizer=optimizer)

    sess = ed.get_session()
    qy_mu_value    = sess[:run](qy_features_mu_param)
    qy_sigma_value = sess[:run](qy_features_sigma_param)

    # reset session and graph to free up memory
    tf.reset_default_graph()
    old_sess = ed.get_session()
    old_sess[:close]()
    ed.util[:graphs][:_ED_SESSION] = tf.InteractiveSession()

    return qy_mu_value, qy_sigma_value
end


function estimate_expression(likapprox_data, y0, sample_factors)
    num_samples, n = y0[:get_shape]()[:as_list]()
    y, y_mu_param, y_sigma_param, y_mu, likapprox = transcript_quantification_model(likapprox_data, y0)

    println("Estimating...")

    qy_mu_param = tf.Variable(y0)
    qy_sigma_param = tf.identity(tf.Variable(tf.fill([num_samples, n], 0.1)))
    qy = edmodels.MultivariateNormalDiag(qy_mu_param, qy_sigma_param)

    qy_mu_mu_param = tf.Variable(y0[1])
    qy_mu_sigma_param = tf.identity(tf.Variable(tf.fill([n], 0.1)))
    qy_mu = edmodels.MultivariateNormalDiag(qy_mu_mu_param, qy_mu_sigma_param)

    inference = ed.KLqp(PyDict(Dict(y => qy, y_mu => qy_mu)),
                        data=PyDict(Dict(likapprox => likapprox_data)))

    optimizer = tf.train[:AdamOptimizer](1e-2)
    #optimizer = tf.train[:MomentumOptimizer](1e-3, 0.99)
    inference[:run](n_iter=250, optimizer=optimizer)

    sess = ed.get_session()
    qy_mu_value    = sess[:run](qy_mu_param)
    qy_sigma_value = sess[:run](qy_sigma_param)

    # reset session and graph to free up memory
    tf.reset_default_graph()
    old_sess = ed.get_session()
    old_sess[:close]()
    ed.util[:graphs][:_ED_SESSION] = tf.InteractiveSession()

    return qy_mu_value, qy_sigma_value
end


EXTRUDER_MODELS["expression"] = estimate_expression

