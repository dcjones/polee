
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from polee_approx_likelihood import *
from polee_training import *


"""
If x is a tensor giving transcript expression, return a tensor giving
the corresponding splicing log-ratios.
"""
def transcript_expression_to_splicing_log_ratios(
        num_features, n, feature_indices, antifeature_indices, x):

    feature_matrix = tf.SparseTensor(
        indices=feature_indices,
        values=tf.ones(np.shape(feature_indices)[0]),
        dense_shape=[num_features, n])

    antifeature_matrix = tf.SparseTensor(
        indices=antifeature_indices,
        values=tf.ones(np.shape(antifeature_indices)[0]),
        dense_shape=[num_features, n])

    splice_lrs = []
    for x_i_ in tf.unstack(x):
        x_i = tf.expand_dims(x_i_, -1)

        feature_expr     = tf.sparse_tensor_dense_matmul(feature_matrix, x_i)
        antifeature_expr = tf.sparse_tensor_dense_matmul(antifeature_matrix, x_i)

        splice_lrs.append(
            tf.log(feature_expr) - tf.log(antifeature_expr))

    return tf.squeeze(tf.stack(splice_lrs), axis=-1)


"""
Approximate likelihood function for splicing ratios using a normal distribution
by minimizing KL(p||q) (not KL(q||p), which is more typical in VI).
"""
def approximate_splicing_likelihood(
        init_feed_dict, vars, num_samples, num_features, n,
        feature_indices, antifeature_indices, sess=None):

    qx_feature_loc = tf.Variable(
        tf.zeros([num_samples, num_features]),
        name="qx_feature_loc")

    qx_feature_scale = tf.nn.softplus(tf.Variable(
        # tf.zeros([num_samples, num_features]),
        tf.fill([num_samples, num_features], -2.0),
        name="qx_feature_scale_softminus"))

    qx_feature = tfd.Normal(
        loc=qx_feature_loc,
        scale=qx_feature_scale)

    x = rnaseq_approx_likelihood_sampler_from_vars(num_samples, n, vars)
    x_feature = transcript_expression_to_splicing_log_ratios(
        num_features, n, feature_indices, antifeature_indices, x)

    log_prob = tf.reduce_sum(qx_feature.log_prob(x_feature))

    if sess is None:
        sess = tf.Session()

    train(sess, -log_prob, init_feed_dict, 500, 1e-2)

    return (sess.run(qx_feature_loc), sess.run(qx_feature_scale))
