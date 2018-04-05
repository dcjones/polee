
# Models explicitly considering splicing rates. Splicing a particularly tricky
# issue due to the weirdness of the transcript expression -> splicing
# transformation. We try to overcome this by building an approximation to the
# splicing rate likelihood function and basing all our models on that.

"""
Build tensorflow transformation from expression vectors to splicing log-ratios.
"""
function transcript_expression_to_splicing_log_ratios(
                num_features, n,
                feature_idxs, feature_transcript_idxs,
                antifeature_idxs, antifeature_transcript_idxs, x)

    feature_matrix = tf.SparseTensor(
        indices=hcat(feature_idxs .- 1, feature_transcript_idxs .- 1),
        values=tf.ones(length(feature_idxs)),
        dense_shape=[num_features, n])

    antifeature_matrix = tf.SparseTensor(
        indices=hcat(antifeature_idxs .- 1, antifeature_transcript_idxs .- 1),
        values=tf.ones(length(antifeature_idxs)),
        dense_shape=[num_features, n])

    splice_lrs = Any[]
    for x_i in tf.unstack(x)
        x_i_ = tf.expand_dims(x_i, -1)

        feature_expr     = tf.sparse_tensor_dense_matmul(feature_matrix, x_i_)
        antifeature_expr = tf.sparse_tensor_dense_matmul(antifeature_matrix, x_i_)

        push!(splice_lrs, tf.log(feature_expr) - tf.log(antifeature_expr))
    end

    splice_lr = tf.squeeze(tf.stack(splice_lrs), axis=-1)
    return splice_lr
end


"""
There isn't a tractable way to evaluate P(reads|splicing ratios), however we
can efficiently sample in proportion to it. We use that fact to build a
Logit-Normal approximation to the splicing likelihood by minimizing KL(p,q)
(not KL(q,p) as is more common in variational inference). This is pretty
simple to do.

    argmin KL(p,q) = argmin -E[log q(x)] + E[log p(x)]

        with expectations wrt to p

    = argmax E[log q(x)]

So all we have to do is optimize q's parameters using stochastic gradient descent
against samples drawn from p.
"""
function approximate_splicing_likelihood(input::ModelInput)
    num_samples, n = size(input.loaded_samples.x0_values)

    (num_features,
     feature_idxs, feature_transcript_idxs,
     antifeature_idxs, antifeature_transcript_idxs) = splicing_features(input)

    # Approximation

    x_feature_loc = rnaseq_approx_likelihood.ImproperPrior(
        value=tf.zeros([num_samples, num_features]))

    x_feature_scale = rnaseq_approx_likelihood.ImproperPrior(
        value=tf.zeros([num_samples, num_features]))

    x_feature = edmodels.NormalWithSoftplusScale(
        loc=x_feature_loc, scale=x_feature_scale)

    # Inference

    qx_feature_loc = edmodels.PointMass(
        tf.Variable(tf.zeros([num_samples, num_features]),
                    name="qx_feature_loc_param"))
    qx_feature_scale = edmodels.PointMass(
        tf.Variable(tf.zeros([num_samples, num_features]),
                    name="qx_feature_scale_param"))

    T = x -> transcript_expression_to_splicing_log_ratios(
                num_features, n, feature_idxs, feature_transcript_idxs,
                antifeature_idxs, antifeature_transcript_idxs, x)

    optimizer = tf.train[:AdamOptimizer](5e-2)
    latent_vars = Dict(
        x_feature_loc    => qx_feature_loc,
        x_feature_scale  => qx_feature_scale
    )
    run_implicit_model_map_inference(input, x_feature, T, latent_vars, 500, optimizer)

    sess = ed.get_session()
    return (sess[:run](qx_feature_loc), sess[:run](qx_feature_scale))
end

