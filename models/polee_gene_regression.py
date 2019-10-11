

"""
Simple bayesian linear regression at the gene level.

This is largely the same as transcript linear regression but with model
at the gene level, and extra parameters representing inter-gene transcript
mixture.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import sys
from tensorflow_probability import edward2 as ed
import polee
from polee_approx_likelihood import *
from polee_training import *
from polee_regression import *


"""
Define model for linear regression.
    * `num_factors`: Number of factors
    * `num_features`: Number of genes
    * `n`: Number of transcripts
    * `F`: 0/1 design matrix of shape [num_samples, num_factors]
"""
def linear_gene_regression_model(
        num_factors, num_features, n, F,
        x_bias_loc, x_bias_scale, x_scale_hinges, sample_scales):

    num_samples = len(sample_scales)

    w_global_scale_variance, w_global_scale_noncentered, \
        w_local_scale_variance, w_local_scale_noncentered, \
        w, x_bias, x_scale_concentration_c, x_scale_rate_c, x_scale, \
        w_distortion_c, x_gene = linear_regression_model(
            num_factors, num_features, F, x_bias_loc, x_bias_scale, x_scale_hinges, sample_scales)

    x_isoform = ed.Normal(
        loc=tf.zeros([num_samples, n]),
        scale=10.0,
        name="x_isoform")

    return w_global_scale_variance, w_global_scale_noncentered, \
        w_local_scale_variance, w_local_scale_noncentered, \
        w, x_bias, x_scale_concentration_c, x_scale_rate_c, x_scale, \
        w_distortion_c, x_gene, x_isoform


"""
Variational model for linear regression, to be paired with `linear_regression_model`.
"""
def linear_gene_regression_variational_model(
        qw_global_scale_variance_loc_var, qw_global_scale_variance_scale_var,
        qw_global_scale_noncentered_loc_var, qw_global_scale_noncentered_scale_var,
        qw_local_scale_variance_loc_var, qw_local_scale_variance_scale_var,
        qw_local_scale_noncentered_loc_var, qw_local_scale_noncentered_scale_var,
        qw_loc_var, qw_scale_var,
        qx_bias_loc_var, qx_bias_scale_var,
        qx_scale_concentration_c_loc_var,
        qx_scale_rate_c_loc_var,
        qx_scale_loc_var, qx_scale_scale_var,
        qw_distortion_c_loc_var,
        qx_gene_loc_var, qx_gene_scale_var,
        qx_isoform_loc_var, qx_isoform_scale_var,
        use_point_estimates):

    qw_global_scale_variance, qw_global_scale_noncentered, \
        qw_local_scale_variance, qw_local_scale_noncentered, qw, qx_bias, \
        qx_scale_concentration_c, qx_scale_rate_c, qx_scale, \
        qw_distortion_c, qx_gene = linear_regression_variational_model(
            qw_global_scale_variance_loc_var, qw_global_scale_variance_scale_var,
            qw_global_scale_noncentered_loc_var, qw_global_scale_noncentered_scale_var,
            qw_local_scale_variance_loc_var, qw_local_scale_variance_scale_var,
            qw_local_scale_noncentered_loc_var, qw_local_scale_noncentered_scale_var,
            qw_loc_var, qw_scale_var,
            qx_bias_loc_var, qx_bias_scale_var,
            qx_scale_concentration_c_loc_var,
            qx_scale_rate_c_loc_var,
            qx_scale_loc_var, qx_scale_scale_var,
            qw_distortion_c_loc_var,
            qx_gene_loc_var, qx_gene_scale_var,
            use_point_estimates)

    qx_isoform = ed.Normal(
        loc=qx_isoform_loc_var,
        scale=qx_isoform_scale_var,
        name="qx_isoform")

    return qw_global_scale_variance, qw_global_scale_noncentered, \
        qw_local_scale_variance, qw_local_scale_noncentered, qw, qx_bias, \
        qx_scale_concentration_c, qx_scale_rate_c, qx_scale, \
        qw_distortion_c, qx_gene, qx_isoform


"""
Set up a linear regression model for variational inference, returning
"""
def linear_gene_regression_inference(
        init_feed_dict, n, F, x_gene_init, x_isoform_init, make_likelihood,
        x_bias_mu0, x_bias_sigma0, x_scale_hinges, sample_scales,
        use_point_estimates, sess, niter=20000, elbo_add_term=0.0):

    num_samples = int(F.shape[0])
    num_factors = int(F.shape[1])
    num_features = int(x_gene_init.shape[1])

    log_joint = ed.make_log_joint_fn(
        lambda: linear_gene_regression_model
            (num_factors, num_features, n, F, x_bias_mu0, x_bias_sigma0, x_scale_hinges, sample_scales))

    qw_global_scale_variance_loc_var = tf.Variable(0.0, name="qw_global_scale_variance_loc_var")
    qw_global_scale_variance_scale_var = tf.nn.softplus(tf.Variable(-1.0, name="qw_global_scale_variance_scale_var"))

    qw_global_scale_noncentered_loc_var = tf.Variable(0.0, name="qw_global_scale_noncentered_loc_var")
    qw_global_scale_noncentered_scale_var = tf.nn.softplus(tf.Variable(-1.0, name="qw_global_scale_noncentered_scale_var"))

    qw_local_scale_variance_loc_var = tf.Variable(
        tf.fill([num_features], 0.0),
        name="qw_local_scale_variance_loc_var")
    qw_local_scale_variance_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_features], -1.0),
        name="qw_local_scale_variance_scale_var"))

    qw_local_scale_noncentered_loc_var = tf.Variable(
        tf.fill([num_features], 0.0),
        name="qw_local_scale_noncentered_loc_var")
    qw_local_scale_noncentered_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_features], -1.0),
        name="qw_local_scale_noncentered_scale_var"))

    qw_loc_var = tf.Variable(
        tf.zeros([num_features, num_factors]),
        name="qw_loc_var")
    qw_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_features, num_factors], -2.0),
        name="qw_scale_var"))

    qx_bias_loc_var = tf.Variable(
        tf.reduce_mean(x_gene_init, axis=0),
        name="qx_bias_loc_var")
    qx_bias_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_features], -1.0),
        name="qx_bias_scale_var"))

    qx_scale_concentration_c_loc_var = tf.Variable(
        tf.fill([kernel_regression_degree], 1.0),
        name="qx_scale_concentration_c_loc_var")

    qx_scale_rate_c_loc_var = tf.Variable(
        tf.fill([kernel_regression_degree], -15.0),
        name="qx_scale_rate_c_loc_var")

    qx_scale_loc_var = tf.Variable(
        tf.fill([num_features], -0.5),
        name="qx_scale_loc_var")
    qx_scale_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_features], -1.0),
        name="qx_scale_scale_var"))

    qw_distortion_c_loc_var = tf.Variable(
        tf.zeros([num_factors, kernel_regression_degree]),
        name="qw_distortion_c_loc_var")

    qx_gene_loc_var = tf.Variable(
        x_gene_init,
        name="qx_gene_loc_var",
        trainable=not use_point_estimates)

    qx_gene_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_samples, num_features], -1.0),
        name="qx_gene_scale_var"))

    qx_isoform_loc_var = tf.Variable(
        x_isoform_init,
        name="qx_isoform_loc_var",
        trainable=not use_point_estimates)

    qx_isoform_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_samples, n], -1.0),
        name="qx_isoform_scale_var"))

    qw_global_scale_variance, qw_global_scale_noncentered, \
        qw_local_scale_variance, qw_local_scale_noncentered, qw, qx_bias, \
        qx_scale_concentration_c, qx_scale_rate_c, qx_scale, \
        qw_distortion_c, qx_gene, qx_isoform = \
        linear_gene_regression_variational_model(
            qw_global_scale_variance_loc_var,    qw_global_scale_variance_scale_var,
            qw_global_scale_noncentered_loc_var, qw_global_scale_noncentered_scale_var,
            qw_local_scale_variance_loc_var,     qw_local_scale_variance_scale_var,
            qw_local_scale_noncentered_loc_var,  qw_local_scale_noncentered_scale_var,
            qw_loc_var,                          qw_scale_var,
            qx_bias_loc_var,                     qx_bias_scale_var,
            qx_scale_concentration_c_loc_var,
            qx_scale_rate_c_loc_var,
            qx_scale_loc_var,                    qx_scale_scale_var,
            qw_distortion_c_loc_var,
            qx_gene_loc_var,                     qx_gene_scale_var,
            qx_isoform_loc_var,                  qx_isoform_scale_var,
            use_point_estimates)

    log_prior = log_joint(
        w_global_scale_variance=qw_global_scale_variance,
        w_global_scale_noncentered=qw_global_scale_noncentered,
        w_local_scale_variance=qw_local_scale_variance,
        w_local_scale_noncentered=qw_local_scale_noncentered,
        w=qw,
        x_bias=qx_bias,
        x_scale_concentration_c=qx_scale_concentration_c,
        x_scale_rate_c=qx_scale_rate_c,
        x_scale=qx_scale,
        w_distortion_c=qw_distortion_c,
        x=qx_gene,
        x_isoform=qx_isoform)

    variational_log_joint = ed.make_log_joint_fn(
        lambda: linear_gene_regression_variational_model(
            qw_global_scale_variance_loc_var,    qw_global_scale_variance_scale_var,
            qw_global_scale_noncentered_loc_var, qw_global_scale_noncentered_scale_var,
            qw_local_scale_variance_loc_var,     qw_local_scale_variance_scale_var,
            qw_local_scale_noncentered_loc_var,  qw_local_scale_noncentered_scale_var,
            qw_loc_var,                          qw_scale_var,
            qx_bias_loc_var,                     qx_bias_scale_var,
            qx_scale_concentration_c_loc_var,
            qx_scale_rate_c_loc_var,
            qx_scale_loc_var,                    qx_scale_scale_var,
            qw_distortion_c_loc_var,
            qx_gene_loc_var,                     qx_gene_scale_var,
            qx_isoform_loc_var,                  qx_isoform_scale_var,
            use_point_estimates))

    entropy = variational_log_joint(
        qw_global_scale_variance=qw_global_scale_variance,
        qw_global_scale_noncentered=qw_global_scale_noncentered,
        qw_local_scale_variance=qw_local_scale_variance,
        qw_local_scale_noncentered=qw_local_scale_noncentered,
        qw=qw,
        qx_bias=qx_bias,
        qx_scale=qx_scale,
        qx_scale_concentration_c=qx_scale_concentration_c,
        qx_scale_rate_c=qx_scale_rate_c,
        qw_distortion_c=qw_distortion_c,
        qx=qx_gene,
        qx_isoform=qx_isoform)

    log_likelihood = make_likelihood(qx_gene, qx_isoform)

    scale_penalty = tf.reduce_sum(tfd.Normal(loc=0.0, scale=5e-4).log_prob(
        tf.log(tf.reduce_sum(tf.exp(qx_gene), axis=1))))

    elbo = log_prior + log_likelihood - entropy + elbo_add_term + scale_penalty
    elbo = tf.check_numerics(elbo, "Non-finite ELBO value")

    if sess is None:
        sess = tf.Session()

    train(sess, -elbo, init_feed_dict, niter, 1e-3, decay_rate=1.0)
    # train(sess, -elbo, init_feed_dict, niter, 1e-2, decay_rate=0.995)

    return (
        sess.run(qx_gene.distribution.mean()),
        sess.run(qw.distribution.mean()),
        sess.run(qw.distribution.stddev()),
        sess.run(qx_bias.distribution.mean()),
        sess.run(qx_scale))


def gene_log_likelihood(
        vars, n, num_features, feature_idxs, transcript_idxs, x_gene, x_isoform):
    x_isoform_indices = np.empty([n, 2], dtype=np.int)
    for (k, (i, j)) in enumerate(zip(feature_idxs, transcript_idxs)):
        x_isoform_indices[k, 0] = i - 1
        x_isoform_indices[k, 1] = j - 1

    xs = []
    for (x_gene_i, x_isoform_i) in zip(tf.unstack(x_gene), tf.unstack(x_isoform)):
        x_isoform_matrix = tf.SparseTensor(
            indices=x_isoform_indices,
            values=x_isoform_i,
            dense_shape=[num_features, n])
        x_isoform_matrix_softmax = tf.sparse_softmax(x_isoform_matrix)
        x_i_exp = tf.sparse_tensor_dense_matmul(
            x_isoform_matrix_softmax,
            tf.expand_dims(tf.exp(x_gene_i), -1),
            adjoint_a=True)
        x_i_exp = tf.clip_by_value(x_i_exp, 1e-16, np.inf)
        x_i = tf.log(x_i_exp)
        xs.append(x_i)
    x = tf.squeeze(tf.stack(xs), axis=-1)

    x = tf.Print(x, [tf.reduce_sum(tf.exp(x), axis=1)], "x scales", summarize=6)
    x = tf.Print(x, [tf.reduce_sum(tf.exp(x_gene), axis=1)], "x_gene scales", summarize=6)

    gene_sizes = np.zeros(num_features, dtype=np.float32)
    for i in feature_idxs:
        gene_sizes[i-1] += 1

    gene_prior_adjustment = polee.noninformative_gene_prior(
        tf.log(tf.nn.softmax(x_gene, axis=1)), gene_sizes)

    return rnaseq_approx_likelihood_from_vars(vars, x) + gene_prior_adjustment


def estimate_gene_linear_regression(
        init_feed_dict, vars, n, num_features,
        feature_idxs, transcript_idxs, x_gene_init, x_isoform_init, F_arr,
        sample_scales, use_point_estimates, sess=None):

    F = tf.constant(F_arr, dtype=tf.float32)

    x_bias_mu0 = np.log(1/num_features)
    x_bias_sigma0 = 12.0

    make_likelihood = lambda x_gene, x_isoform: \
        gene_log_likelihood(
            vars, n, num_features, feature_idxs,
            transcript_idxs, x_gene, x_isoform)

    x_scale_hinges = choose_knots(np.min(x_gene_init), np.max(x_gene_init))

    return linear_gene_regression_inference(
        init_feed_dict, n, F, x_gene_init, x_isoform_init, make_likelihood,
        x_bias_mu0, x_bias_sigma0, x_scale_hinges, sample_scales,
        use_point_estimates, sess, niter=8000)
