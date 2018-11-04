
"""
Simple bayesian linear regression.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import sys
from tensorflow_probability import edward2 as ed
from polee_approx_likelihood import *
from polee_training import *
from polee_approx_likelihood import *


def transcript_linear_regression_model(num_factors, n, X):
    w_mu0 = 0.0
    w_sigma0 = 0.5
    w_bias_mu0 = np.log(1/n)
    w_bias_sigma0 = 4.0

    sigma_alpha0 = 0.001
    sigma_beta0 = 0.001

    x_df0 = 10.0

    w_sigma = tf.concat(
        [tf.constant(w_bias_sigma0, shape=[1, n], dtype=tf.float32),
         tf.constant(w_sigma0, shape=[num_factors-1, n], dtype=tf.float32)], 0)
    w_mu = tf.concat(
        [tf.constant(w_bias_mu0, shape=[1, n], dtype=tf.float32),
         tf.constant(w_mu0, shape=[num_factors-1, n], dtype=tf.float32)], 0)

    w = ed.Normal(
        loc=w_mu,
        scale=w_sigma,
        name="w")

    x_mu = tf.matmul(X, w, name="x_mu")

    x_sigma_alpha0 = tf.constant(sigma_alpha0, shape=[n], dtype=tf.float32)
    x_sigma_beta0  = tf.constant(sigma_beta0, shape=[n], dtype=tf.float32)
    x_sigma_sq = ed.InverseGamma(
        concentration=x_sigma_alpha0,
        rate=x_sigma_beta0,
        name="x_sigma_sq")
    x_sigma = tf.sqrt(x_sigma_sq, name="x_sigma")

    x = ed.StudentT(
        df=x_df0,
        loc=x_mu,
        scale=x_sigma,
        name="x")

    return w, x_sigma_sq, x


def transcript_linear_regression_variational_model(
        qw_loc, qw_scale,
        qx_sigma_sq_loc, qx_sigma_sq_scale,
        qx_loc, qx_scale):

    qw = ed.Normal(
        loc=qw_loc,
        scale=qw_scale,
        name="qw")

    qx_sigma_sq = ed.TransformedDistribution(
        distribution=
            tfp.distributions.Normal(
                loc=qx_sigma_sq_loc,
                scale=qx_sigma_sq_scale),
        bijector=tfp.bijectors.Exp(),
        name="qx_sigma_sq")

    qx = ed.Normal(
        loc=qx_loc,
        scale=qx_scale,
        name="qx")

    return qw, qx_sigma_sq, qx


def estimate_transcript_linear_regression(
        init_feed_dict, num_samples, n, vars, x0_log, X_arr, sess=None):

    X = tf.constant(X_arr, dtype=tf.float32)
    num_factors = X.shape[1]

    log_joint = ed.make_log_joint_fn(
        lambda: transcript_linear_regression_model(num_factors, n, X))

    qw_loc_init = np.vstack(
        [np.mean(x0_log, 0),
         np.zeros((num_factors - 1, n), np.float32)])

    qw_loc = tf.Variable(
        qw_loc_init,
        name="qw_loc",
        dtype=tf.float32)
    qw_scale = tf.nn.softplus(tf.Variable(
        tf.fill([num_factors, n], -2.0),
        name="qw_softminus_scale",
        dtype=tf.float32))

    qx_sigma_sq_loc = tf.Variable(
        tf.fill([n], 0.0),
        name="qx_sigma_sq_mu_param",
        dtype=tf.float32)
    qx_sigma_sq_scale = tf.nn.softplus(tf.Variable(
        tf.fill([n], 1.0),
        name="qx_sigma_sq_softminus_scale",
        dtype=tf.float32))

    qx_loc = tf.Variable(
        x0_log,
        name="qx_loc",
        dtype=tf.float32)
    qx_scale = tf.nn.softplus(tf.Variable(
        tf.fill([num_samples, n], -2.0),
        name="qx_softminus_scale",
        dtype=tf.float32))

    qw, qx_sigma_sq, qx = transcript_linear_regression_variational_model(
        qw_loc, qw_scale, qx_sigma_sq_loc, qx_sigma_sq_scale, qx_loc, qx_scale)

    lp = log_joint(
        w=qw,
        x_sigma_sq=qx_sigma_sq,
        x=qx)

    variational_log_joint = ed.make_log_joint_fn(
        lambda: transcript_linear_regression_variational_model(
            qw_loc, qw_scale, qx_sigma_sq_loc, qx_sigma_sq_scale, qx_loc, qx_scale))

    entropy = variational_log_joint(
        qw=qw,
        qx_sigma_sq=qx_sigma_sq,
        qx=qx)

    likelihood = rnaseq_approx_likelihood_from_vars(vars, qx)

    elbo = lp + likelihood - entropy

    if sess is None:
        sess = tf.Session()
    train(sess, -elbo, init_feed_dict, 500, 2e-2)

    # point estimate of x error by taking exponent of log-normal mean.
    x_sigma_sq_mean = sess.run(tf.sqrt(tf.exp(
        tf.add(qx_sigma_sq_loc,
            tf.div(tf.square(qx_sigma_sq_scale), 2.0)))))

    return (sess.run(qw.distribution.loc),
        sess.run(qw.distribution.scale),
        x_sigma_sq_mean)
