import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from polee_approx_likelihood import *
from polee_training import *


"""
Compute gene expression from transcript expression (x).

`feature_idxs` and `transcript_idxs` are two vectors of equal length assigning
transcripts to features.
"""
def transcript_expression_to_feature_expression(
        num_features, n, feature_idxs, transcript_idxs, x):

    indices = np.transpose(np.vstack([feature_idxs, transcript_idxs]))

    feature_matrix = tf.SparseTensor(
        indices=indices,
        values=tf.ones(indices.shape[0], tf.float32),
        dense_shape=[num_features, n])

    return tf.transpose(tf.sparse_tensor_dense_matmul(
        feature_matrix, x, adjoint_b=True, name="x_feature"))


"""
Approximate likelihood function for features (typically gene), where a
"feature" is a set of transcripts. Approximated using a normal distribution
and minimizing KL(p||q), where `p` is the "true" distribution (not really
true, since it itself is Polya tree approximation).

This is very similar to what we do with splicing likelihood.
"""
def approximate_feature_likelihood(
        init_feed_dict, vars, num_samples, num_features, n,
        feature_idxs, transcript_idxs, sess=None):

    qx_feature_loc = tf.Variable(
        tf.fill([num_samples, num_features], np.float32(np.log(1/num_features))),
        name="qx_feature_loc")

    qx_feature_scale = tf.nn.softplus(tf.Variable(
        tf.fill([num_samples, num_features], np.float32(-2.0)),
        name="qx_feature_scale_softminus"))

    qx_feature = tfd.Normal(
        loc=qx_feature_loc,
        scale=qx_feature_scale)

    x = rnaseq_approx_likelihood_sampler_from_vars(num_samples, n, vars)
    x_feature = tf.log(transcript_expression_to_feature_expression(
        num_features, n, feature_idxs, transcript_idxs, x))

    log_prob = tf.reduce_sum(qx_feature.log_prob(x_feature))

    if sess is None:
        sess = tf.Session()

    train(sess, -log_prob, init_feed_dict, 1500, 1e-1)

    return (sess.run(qx_feature_loc), sess.run(qx_feature_scale))
