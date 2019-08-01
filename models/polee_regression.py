
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


scale_spline_degree = 8


"""
Define model for linear regression.
    * `num_factors`: Number of factors
    * `num_features`: Dimensionality
    * `F`: 0/1 design matrix of shape [num_samples, num_factors]
"""
def linear_regression_model(
        num_factors, num_features, F,
        x_bias_loc, x_bias_scale, x_scale_hinges):

    x_bias = ed.Normal(
        loc=tf.fill([num_features], np.float32(x_bias_loc)),
        scale=np.float32(x_bias_scale),
        name="x_bias")

    x_scale_hinges_diff = 1.5 * tf.square(tf.expand_dims(x_bias, 0) - tf.expand_dims(x_scale_hinges, -1))
    x_scale_hinges_weight = tf.exp(-x_scale_hinges_diff) # [scale_spline_degree, num_features]
    x_scale_hinges_weight = x_scale_hinges_weight / tf.reduce_sum(x_scale_hinges_weight, axis=0, keepdims=True)

    # w
    # -

    # horseshoe prior
    tau = ed.HalfCauchy(loc=tf.zeros([scale_spline_degree]), scale=10.0, name="tau")

    # [num_features]
    tau_mix = tf.reduce_sum(
        tf.expand_dims(tau, -1) * x_scale_hinges_weight, axis=0)

    w = ed.StudentT(
        df=1.0,
        loc=0.0,
        scale=tf.expand_dims(tau_mix, -1),
        name="w")

    # x
    # -

    x_loc = tf.identity(
        tf.matmul(F, w, transpose_b=True) + x_bias,
        name="x_loc")

    x_scale_c = ed.Normal(
        loc=tf.fill([scale_spline_degree], 0.0), scale=100.0, name="x_scale_c")

    x_scale_loc_mix = tf.reduce_sum(
        tf.expand_dims(x_scale_c, -1) * x_scale_hinges_weight, axis=0)

    x_scale = ed.TransformedDistribution(
        distribution=tfd.StudentT(
            df=1.0,
            loc=x_scale_loc_mix,
            scale=0.5),
        bijector=tfp.bijectors.Exp(),
        name="x_scale")

    x = ed.Normal(
        loc=x_loc,
        scale=x_scale,
        name="x")

    return tau, w, x_bias, x_scale_c, x_scale, x


"""
Variational model for linear regression, to be paired with `linear_regression_model`.
"""
def linear_regression_variational_model(
        qtau_loc_var, qtau_scale_var,
        qw_loc_var, qw_scale_var,
        qx_bias_loc_var, qx_bias_scale_var,
        qx_scale_c_loc_var, qx_scale_c_scale_var,
        qx_scale_loc_var, qx_scale_scale_var,
        qx_loc_var, qx_scale_var,
        use_point_estimates):

    qtau = ed.Deterministic(
        loc=tf.exp(qtau_loc_var),
        name="qtau")

    qw = ed.Normal(
        loc=qw_loc_var,
        scale=qw_scale_var,
        name="qw")

    qx_bias = ed.Normal(
        loc=qx_bias_loc_var,
        scale=qx_bias_scale_var,
        name="qx_bias")

    qx_scale_c = ed.Deterministic(
        loc=qx_scale_c_loc_var,
        name="qx_scale_c")

    qx_scale = ed.LogNormal(
        loc=qx_scale_loc_var,
        scale=qx_scale_scale_var,
        name="qx_scale")

    if use_point_estimates:
        qx = ed.Deterministic(loc=qx_loc_var, name="qx")
    else:
        qx = ed.Normal(
            loc=qx_loc_var,
            scale=qx_scale_var,
            name="qx")

    return qtau, qw, qx_bias, qx_scale_c, qx_scale, qx



"""
Set up a linear regression model for variational inference, returning
"""
def linear_regression_inference(
        init_feed_dict, F, x_init, make_likelihood,
        x_bias_mu0, x_bias_sigma0, x_scale_hinges, use_point_estimates, sess):

    num_samples = int(F.shape[0])
    num_factors = int(F.shape[1])
    num_features = int(x_init.shape[1])

    log_joint = ed.make_log_joint_fn(
        lambda: linear_regression_model
            (num_factors, num_features, F, x_bias_mu0, x_bias_sigma0, x_scale_hinges))

    qtau_loc_var = tf.Variable(tf.zeros([scale_spline_degree]), name="qtau_loc_var")
    qtau_scale_var = tf.nn.softplus(tf.Variable(tf.zeros([scale_spline_degree]), name="qtau_scale_var"))

    qw_loc_var = tf.Variable(
        tf.zeros([num_features, num_factors]),
        name="qx_loc_var")
    qw_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_features, num_factors], -1.0),
        name="qx_scale_var"))

    qw_loc_var = tf.check_numerics(qw_loc_var, "Non-finite value in qw_loc_var")
    qw_scale_var = tf.check_numerics(qw_scale_var, "Non-finite value in qw_scale_var")

    qx_bias_loc_var = tf.Variable(
        tf.reduce_mean(x_init, axis=0),
        name="qx_bias_loc_var")
    qx_bias_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_features], -1.0),
        name="qx_bias_scale_var"))

    qx_scale_c_loc_var = tf.Variable(
        tf.zeros([scale_spline_degree]),
        name="qx_scale_c_loc_var")
    qx_scale_c_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([scale_spline_degree], -1.0),
        name="qx_scale_c_scale_var"))

    qx_scale_loc_var = tf.Variable(
        tf.fill([num_features], 1.0),
        name="qx_scale_loc_var")
    qx_scale_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_features], -1.0),
        name="qx_scale_scale_var"))

    qx_loc_var = tf.Variable(
        x_init,
        name="qx_loc_var",
        trainable=not use_point_estimates)

    qx_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_samples, num_features], 0.0),
        name="qx_scale_var"))

    qtau, qw, qx_bias, qx_scale_c, qx_scale, qx = \
        linear_regression_variational_model(
            qtau_loc_var, qtau_scale_var,
            qw_loc_var, qw_scale_var,
            qx_bias_loc_var, qx_bias_scale_var,
            qx_scale_c_loc_var, qx_scale_c_scale_var,
            qx_scale_loc_var, qx_scale_scale_var,
            qx_loc_var, qx_scale_var,
            use_point_estimates)

    log_prior = log_joint(
        tau=qtau,
        w=qw,
        x_bias=qx_bias,
        x_scale_c=qx_scale_c,
        x_scale=qx_scale,
        x=qx)

    variational_log_joint = ed.make_log_joint_fn(
        lambda: linear_regression_variational_model(
            qtau_loc_var, qtau_scale_var,
            qw_loc_var, qw_scale_var,
            qx_bias_loc_var, qx_bias_scale_var,
            qx_scale_c_loc_var, qx_scale_c_scale_var,
            qx_scale_loc_var, qx_scale_scale_var,
            qx_loc_var, qx_scale_var,
            use_point_estimates))

    entropy = variational_log_joint(
        qtau=qtau,
        qw=qw,
        qx_bias=qx_bias,
        qx_scale=qx_scale,
        qx_scale_c=qx_scale_c,
        qx=qx)

    log_likelihood = make_likelihood(qx)

    elbo = log_prior + log_likelihood - entropy

    if sess is None:
        sess = tf.Session()

    train(sess, -elbo, init_feed_dict, 20000, 1e-3, decay_rate=0.999)

    return (
        sess.run(qx.distribution.mean()),
        sess.run(qw.distribution.mean()),
        sess.run(qw.distribution.stddev()),
        sess.run(qx_bias.distribution.mean()),
        sess.run(qx_scale.distribution.mean()))


"""
Run variational inference on transcript expression linear regression.
"""
def estimate_transcript_linear_regression(
        init_feed_dict, vars, x_init, F_arr,
        use_point_estimates, sess=None):

    F = tf.constant(F_arr, dtype=tf.float32)
    num_features = x_init.shape[1]

    # TODO: trying different initalization
    x_init = np.repeat(np.mean(x_init, axis=0, keepdims=True), repeats=x_init.shape[0], axis=0)

    x_bias_mu0 = np.log(1/num_features)
    x_bias_sigma0 = 16.0

    if use_point_estimates:
        make_likelihood = lambda qx: 0.0
    else:
        make_likelihood = lambda qx: rnaseq_approx_likelihood_from_vars(vars, qx)

    return linear_regression_inference(
        init_feed_dict, F, x_init, make_likelihood,
        x_bias_mu0, x_bias_sigma0, use_point_estimates, sess)


"""
Run variational inference on feature log-ratio linear regression.
"""
def estimate_feature_linear_regression(
        init_feed_dict, feature_loc, feature_scale, x0_log, F_arr,
        use_point_estimates, sess=None):

    # don't need this since we already used the transcript expression
    # likelihood approximation to approximate splicing likelihood.
    init_feed_dict.clear()

    num_samples = feature_loc.shape[0]
    num_features = feature_scale.shape[1]

    feature_likelihood = ed.Normal(
        loc=feature_loc,
        scale=feature_scale,
        name="feature_likelihood")

    make_likelihood = lambda qx: tf.reduce_sum(feature_likelihood.distribution.log_prob(
        tf.log(tf.nn.softmax(qx, axis=1))))

    F = tf.constant(F_arr, dtype=tf.float32)

    x_init = tf.log(tf.nn.softmax(feature_loc, axis=1))

    # choose equally spaced points for piecewise variance function
    x_init_exp = np.exp(feature_loc)
    x_init_mean = np.mean(np.log(x_init_exp / np.sum(x_init_exp, axis=1, keepdims=True)), axis=0)

    x_scale_hinges = []
    x_init_mean_min = np.min(x_init_mean)
    x_init_mean_max = np.max(x_init_mean)
    d = (x_init_mean_max - x_init_mean_min) / (scale_spline_degree+1)
    for i in range(scale_spline_degree):
        x_scale_hinges.append(x_init_mean_min + (i+1)*d)

    print(x_scale_hinges)

    x_bias_mu0 = np.log(1/num_features)
    x_bias_sigma0 = 16.0

    if sess is None:
        sess = tf.Session()

    return linear_regression_inference(
        init_feed_dict, F, x_init, make_likelihood,
        x_bias_mu0, x_bias_sigma0, x_scale_hinges, use_point_estimates, sess)


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
