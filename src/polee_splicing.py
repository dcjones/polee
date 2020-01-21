
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from polee_approx_likelihood import *
from polee_gene_expression import *
from polee_training import *


"""
If x is a tensor giving transcript expression, return a tensor giving
the corresponding splicing log-ratios.
"""
def transcript_expression_to_splicing_log_ratios(
        num_samples, num_features, n,
        feature_indices_stacked, antifeature_indices_stacked, x):

    feature_matrix = tf.SparseTensor(
        indices=feature_indices_stacked,
        values=tf.ones(np.shape(feature_indices_stacked)[0]),
        dense_shape=[num_samples*num_features, num_samples*n])

    antifeature_matrix = tf.SparseTensor(
        indices=antifeature_indices_stacked,
        values=tf.ones(np.shape(antifeature_indices_stacked)[0]),
        dense_shape=[num_samples*num_features, num_samples*n])

    # batch matmul by forming X into a blockwise sparse matrix
    x_flat = tf.reshape(x, [num_samples * n, 1])

    feature_expr_flat = tf.sparse.sparse_dense_matmul(
        feature_matrix, x_flat)

    antifeature_expr_flat = tf.sparse.sparse_dense_matmul(
        antifeature_matrix, x_flat)

    x_log_ratio_flat = \
        tf.math.log(feature_expr_flat) - tf.math.log(antifeature_expr_flat)

    x_log_ratio = tf.reshape(x_log_ratio_flat, [num_samples, num_features])

    return x_log_ratio


# @tf.function
def sample_splice_feature_log_ratios(
        vars, num_samples, num_features, n,
        feature_idxs_stacked, antifeature_indices_stacked):

    x = rnaseq_approx_likelihood_sampler_from_vars(num_samples, n, vars)

    return transcript_expression_to_splicing_log_ratios(
        num_samples, num_features, n,
        feature_idxs_stacked, antifeature_indices_stacked, x)


"""
Approximate likelihood function for splicing ratios using a normal distribution
by minimizing KL(p||q) (not KL(q||p), which is more typical in VI).
"""
def approximate_splicing_likelihood(
        vars, num_samples, num_features, n,
        feature_indices, antifeature_indices):

    # expand indices into blockwise matrices
    num_entries = feature_indices.shape[0]
    feature_indices_stacked = np.empty(
        [num_samples * num_entries, 2], dtype=np.int)
    for k in range(num_samples):
        for i in range(num_entries):
            feature_indices_stacked[k*num_entries+i,0] = k*num_features+feature_indices[i,0]
            feature_indices_stacked[k*num_entries+i,1] = k*n + feature_indices[i,1]

    num_entries = antifeature_indices.shape[0]
    antifeature_indices_stacked = np.empty(
        [num_samples * num_entries, 2], dtype=np.int)
    for k in range(num_samples):
        for i in range(num_entries):
            antifeature_indices_stacked[k*num_entries+i,0] = k*num_features + antifeature_indices[i,0]
            antifeature_indices_stacked[k*num_entries+i,1] = k*n + antifeature_indices[i,1]

    num_mean_est_samples = 1000
    qx_feature_loc = tf.Variable(
        tf.zeros([num_samples, num_features]))

    # estimate mean
    for i in range(num_mean_est_samples):
        qx_feature_loc.assign_add(
            sample_splice_feature_log_ratios(
                vars, num_samples, num_features, n,
                feature_indices_stacked,
                antifeature_indices_stacked))
    qx_feature_loc.assign(qx_feature_loc / num_mean_est_samples)

    # estimate std. dev.
    num_var_est_samples = 1000
    qx_feature_scale = tf.Variable(tf.zeros([num_samples, num_features]))
    for i in range(num_mean_est_samples):
        qx_feature_scale.assign_add(
            tf.square(
                qx_feature_loc - sample_splice_feature_log_ratios(
                    vars, num_samples, num_features, n,
                    feature_indices_stacked,
                    antifeature_indices_stacked)))
    qx_feature_scale.assign(tf.sqrt(qx_feature_scale / num_var_est_samples))

    print("done")

    return (
        qx_feature_loc.numpy(),
        qx_feature_scale.numpy() )


def estimate_splicing_log_ratios(
        init_feed_dict, vars, num_samples, num_features, n,
        feature_indices, antifeature_indices, sess=None, sigma0=2.0):
    if sess is None:
        sess = tf.Session()

    x_likelihood_loc, x_likelihood_scale = approximate_splicing_likelihood(
        init_feed_dict, vars, num_samples, num_features, n,
        feature_indices, antifeature_indices, sess=sess)

    return estimate_feature_expression_from_normal_approx(
        init_feed_dict, vars, num_samples, num_features, n,
        x_likelihood_loc, x_likelihood_scale, sess=sess,
        mu0=0.0, sigma0=sigma0, softmax_x=False)

