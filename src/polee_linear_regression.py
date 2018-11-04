
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


"""
Define model for linear regression.
    * `num_factors`: Number of factors
    * `n`: Dimensionality
    * `X`: 0/1 design matrix of shape [num_samples, num_factors]
"""
def linear_regression_model(
        num_factors, num_features, X,
        w_mu0, w_sigma0, w_bias_mu0, w_bias_sigma0,
        sigma_alpha0=0.001, sigma_beta0=0.001, x_df0=10.0):
    sigma_alpha0 = 0.001
    sigma_beta0 = 0.001

    x_df0 = 10.0

    w_sigma = tf.concat(
        [tf.constant(w_bias_sigma0, shape=[1, num_features], dtype=tf.float32),
         tf.constant(w_sigma0, shape=[num_factors-1, num_features], dtype=tf.float32)], 0)
    w_mu = tf.concat(
        [tf.constant(w_bias_mu0, shape=[1, num_features], dtype=tf.float32),
         tf.constant(w_mu0, shape=[num_factors-1, num_features], dtype=tf.float32)], 0)

    w = ed.Normal(
        loc=w_mu,
        scale=w_sigma,
        name="w")

    x_mu = tf.matmul(X, w, name="x_mu")

    x_sigma_alpha0 = tf.constant(sigma_alpha0, shape=[num_features], dtype=tf.float32)
    x_sigma_beta0  = tf.constant(sigma_beta0, shape=[num_features], dtype=tf.float32)
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


"""
Variational model for linear regression, to be paired with `linear_regression_model`.
"""
def linear_regression_variational_model(
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



"""
Set up a linear regression model for variational inference, returning
"""
def linear_regression_inference(
        init_feed_dict, X, x_init, make_likelihood,
        w_mu0, w_sigma0, w_bias_mu0, w_bias_sigma0):
    num_samples = int(X.shape[0])
    num_factors = int(X.shape[1])
    num_features = int(x_init.shape[1])

    log_joint = ed.make_log_joint_fn(
        lambda: linear_regression_model
            (num_factors, num_features, X, w_mu0, w_sigma0, w_bias_mu0, w_bias_sigma0))

    qw_loc_init = np.vstack(
        [np.mean(x_init, 0),
         np.zeros((num_factors - 1, num_features), np.float32)])

    qw_loc = tf.Variable(
        qw_loc_init,
        name="qw_loc",
        dtype=tf.float32)
    qw_scale = tf.nn.softplus(tf.Variable(
        tf.fill([num_factors, num_features], -2.0),
        name="qw_softminus_scale",
        dtype=tf.float32))

    qx_sigma_sq_loc = tf.Variable(
        tf.fill([num_features], 0.0),
        name="qx_sigma_sq_mu_param",
        dtype=tf.float32)
    qx_sigma_sq_scale = tf.nn.softplus(tf.Variable(
        tf.fill([num_features], 1.0),
        name="qx_sigma_sq_softminus_scale",
        dtype=tf.float32))

    qx_loc = tf.Variable(
        x_init,
        name="qx_loc",
        dtype=tf.float32)
    qx_scale = tf.nn.softplus(tf.Variable(
        tf.fill([num_samples, num_features], -2.0),
        name="qx_softminus_scale",
        dtype=tf.float32))

    qw, qx_sigma_sq, qx = linear_regression_variational_model(
        qw_loc, qw_scale, qx_sigma_sq_loc, qx_sigma_sq_scale, qx_loc, qx_scale)

    lp = log_joint(
        w=qw,
        x_sigma_sq=qx_sigma_sq,
        x=qx)

    variational_log_joint = ed.make_log_joint_fn(
        lambda: linear_regression_variational_model(
            qw_loc, qw_scale, qx_sigma_sq_loc, qx_sigma_sq_scale, qx_loc, qx_scale))

    entropy = variational_log_joint(
        qw=qw,
        qx_sigma_sq=qx_sigma_sq,
        qx=qx)

    likelihood = make_likelihood(qx)

    elbo = lp + likelihood - entropy

    sess = tf.Session()
    train(sess, -elbo, init_feed_dict, 500, 2e-2)

    # point estimate of x error by taking exponent of log-normal mean.
    x_sigma_sq_mean = sess.run(tf.sqrt(tf.exp(
        tf.add(qx_sigma_sq_loc,
            tf.div(tf.square(qx_sigma_sq_scale), 2.0)))))

    return (sess.run(qw.distribution.loc),
        sess.run(qw.distribution.scale),
        x_sigma_sq_mean)


"""
Run variational inference on transcript expression linear regression.
"""
def estimate_transcript_linear_regression(
        init_feed_dict, vars, x_init, X_arr, sess=None):

    X = tf.constant(X_arr, dtype=tf.float32)
    num_features = x_init.shape[1]

    w_mu0 = 0.0
    w_sigma0 = 0.5
    w_bias_mu0 = np.log(1/num_features)
    w_bias_sigma0 = 4.0

    make_likelihood = lambda qx: rnaseq_approx_likelihood_from_vars(vars, qx)
    return linear_regression_inference(
        init_feed_dict, X, x_init, make_likelihood,
        w_mu0, w_sigma0, w_bias_mu0, w_bias_sigma0)



"""
Run variational inference on splicing log-ratio linear regression.
"""
def estimate_splicing_linear_regression(
        init_feed_dict, splice_lr_loc, splice_lr_scale, x0_log, X_arr, sess=None):

    # don't need this since we already used the transcript expression
    # likelihood approximation to approximate splicing likelihood.
    init_feed_dict.clear()

    num_samples = splice_lr_loc.shape[0]
    num_features = splice_lr_scale.shape[1]

    splice_lr = tfd.Normal(
        loc=splice_lr_loc,
        scale=splice_lr_scale,
        name="splice_lr")

    make_likelihood = lambda qx: tf.reduce_sum(splice_lr.log_prob(qx))

    X = tf.constant(X_arr, dtype=tf.float32)

    # TODO: might try to find a better initialization
    x_init = np.zeros((num_samples, num_features), np.float32)

    w_mu0 = 0.0
    w_sigma0 = 2.0
    w_bias_mu0 = 0.0
    w_bias_sigma0 = 10.0

    return linear_regression_inference(
        init_feed_dict, X, x_init, make_likelihood,
        w_mu0, w_sigma0, w_bias_mu0, w_bias_sigma0)
