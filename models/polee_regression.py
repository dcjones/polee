
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
    * `num_features`: Dimensionality
    * `F`: 0/1 design matrix of shape [num_samples, num_factors]
"""
def linear_regression_model(
        num_factors, num_features, F,
        x_bias_loc, x_bias_scale):

    # w
    # -

    # horseshoe prior
    tau = ed.HalfCauchy(loc=0.0, scale=1.0, name="tau")
    w_scale = ed.HalfCauchy(
        loc=tf.zeros([num_features, num_factors]),
        scale=tau, name="w_scale")

    w = ed.Normal(
        loc=0.0,
        scale=w_scale,
        name="w")

    # x
    # -

    x_bias = ed.Normal(
        loc=tf.fill([num_features], np.float32(x_bias_loc)),
        scale=np.float32(x_bias_scale),
        name="x_bias")

    x_loc = tf.identity(
        tf.matmul(F, w, transpose_b=True) + x_bias,
        name="x_loc")
    print("x_loc")
    print(F)
    print(w)
    print(x_bias)
    print(x_loc)

    x_scale = ed.HalfCauchy(
        loc=tf.fill([num_features], 0.0),
        scale=0.1,
        name="x_scale")

    x = ed.StudentT(
        df=1.0,
        loc=x_loc,
        scale=x_scale,
        name="x")

    return tau, w_scale, w, x_bias, x_scale, x


"""
Variational model for linear regression, to be paired with `linear_regression_model`.
"""
def linear_regression_variational_model(
        qtau_loc_var, qtau_scale_var,
        qw_scale_loc_var, qw_scale_scale_var,
        qw_loc_var, qw_scale_var,
        qx_bias_loc_var, qx_bias_scale_var,
        qx_scale_loc_var, qx_scale_scale_var,
        qx_loc_var, qx_scale_var):

    qtau = ed.LogNormal(
        loc=qtau_loc_var,
        scale=qtau_scale_var,
        name="qtau")

    qw_scale = ed.LogNormal(
        loc=qw_scale_loc_var,
        scale=qw_scale_scale_var,
        name="qw_scale")

    qw = ed.Normal(
        loc=qw_loc_var,
        scale=qw_scale_var,
        name="qw")

    qx_bias = ed.Normal(
        loc=qx_bias_loc_var,
        scale=qx_bias_scale_var,
        name="qx_bias")

    qx_scale = ed.LogNormal(
        loc=qx_scale_loc_var,
        scale=qx_scale_scale_var,
        name="qx_scale")

    qx = ed.Normal(
        loc=qx_loc_var,
        scale=qx_scale_var,
        name="qx")

    return qtau, qw_scale, qw, qx_bias, qx_scale, qx



"""
Set up a linear regression model for variational inference, returning
"""
def linear_regression_inference(
        init_feed_dict, F, x_init, make_likelihood,
        x_bias_mu, x_bias_sigma):
    num_samples = int(F.shape[0])
    num_factors = int(F.shape[1])
    num_features = int(x_init.shape[1])

    log_joint = ed.make_log_joint_fn(
        lambda: linear_regression_model
            (num_factors, num_features, F, x_bias_mu, x_bias_sigma))

    qtau_loc_var = tf.Variable(0.0, name="qtau_loc_var")
    qtau_scale_var = tf.nn.softplus(tf.Variable(0.0, name="qtau_scale_var"))

    qw_scale_loc_var = tf.Variable(
        tf.zeros([num_features, num_factors]),
        name="qx_scale_loc_var")
    qw_scale_scale_var = tf.nn.softplus(tf.Variable(
        tf.zeros([num_features, num_factors]),
        name="qx_scale_scale_var"))

    qw_loc_var = tf.Variable(
        tf.zeros([num_features, num_factors]),
        name="qx_loc_var")
    tf.summary.histogram("qw_loc_var", qw_loc_var)
    qw_scale_var = tf.nn.softplus(tf.Variable(
        tf.zeros([num_features, num_factors]),
        name="qx_scale_var"))

    qx_bias_loc_var = tf.Variable(
        np.mean(x_init, 0),
        name="qx_bias_loc_var")
    qx_bias_scale_var = tf.nn.softplus(tf.Variable(
        tf.zeros([num_features]),
        name="qx_bias_scale_var"))

    qx_scale_loc_var = tf.Variable(
        tf.zeros([num_features]),
        name="qx_scale_loc_var")
    qx_scale_scale_var = tf.nn.softplus(tf.Variable(
        tf.zeros([num_features]),
        name="qx_scale_scale_var"))

    qx_loc_var = tf.Variable(
        x_init,
        name="qx_loc_var")
    qx_scale_var = tf.nn.softplus(tf.Variable(
        tf.zeros([num_samples, num_features]) ,
        name="qx_scale_var"))

    qtau, qw_scale, qw, qx_bias, qx_scale, qx = \
        linear_regression_variational_model(
            qtau_loc_var, qtau_scale_var,
            qw_scale_loc_var, qw_scale_scale_var,
            qw_loc_var, qw_scale_var,
            qx_bias_loc_var, qx_bias_scale_var,
            qx_scale_loc_var, qx_scale_scale_var,
            qx_loc_var, qx_scale_var)

    log_prior = log_joint(
        tau=qtau,
        w_scale=qw_scale,
        w=qw,
        x_bias=qx_bias,
        x_scale=qx_scale,
        x=qx)

    variational_log_joint = ed.make_log_joint_fn(
        lambda: linear_regression_variational_model(
            qtau_loc_var, qtau_scale_var,
            qw_scale_loc_var, qw_scale_scale_var,
            qw_loc_var, qw_scale_var,
            qx_bias_loc_var, qx_bias_scale_var,
            qx_scale_loc_var, qx_scale_scale_var,
            qx_loc_var, qx_scale_var))

    entropy = variational_log_joint(
        qtau=qtau,
        qw_scale=qw_scale,
        qw=qw,
        qx_bias=qx_bias,
        qx_scale=qx_scale,
        qx=qx)

    log_likelihood = make_likelihood(qx)

    elbo = log_prior + log_likelihood - entropy

    tf.summary.scalar("log likelihood", log_likelihood)
    tf.summary.scalar("log prior", log_prior)
    tf.summary.scalar("elbo", elbo)

    sess = tf.Session()
    train(sess, -elbo, init_feed_dict, 500, 2e-2)

    return (
        sess.run(qw_loc_var),
        sess.run(qw.distribution.scale),
        sess.run(qx_scale.distribution.loc))


"""
Run variational inference on transcript expression linear regression.
"""
def estimate_transcript_linear_regression(
        init_feed_dict, vars, x_init, F_arr, sess=None):

    F = tf.constant(F_arr, dtype=tf.float32)
    num_features = x_init.shape[1]

    x_bias_mu0 = np.log(1/num_features)
    x_bias_sigma0 = 8.0

    make_likelihood = lambda qx: rnaseq_approx_likelihood_from_vars(vars, qx)
    return linear_regression_inference(
        init_feed_dict, F, x_init, make_likelihood,
        x_bias_mu0, x_bias_sigma0)


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
    w_sigma0 = 10.0
    w_bias_mu0 = 0.0
    w_bias_sigma0 = 10.0

    return linear_regression_inference(
        init_feed_dict, X, x_init, make_likelihood,
        w_mu0, w_sigma0, w_bias_mu0, w_bias_sigma0)
