
"""
Linear regression with a regression coefficients factored into a low
dimensional space. kind of like doing a regression then PCA on the coefficients,
but in one model.
"""


import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import sys
from tensorflow_probability import edward2 as ed
from polee_approx_likelihood import *
from polee_training import *


scale_spline_degree = 10



"""
Define model for linear regression.
    * `num_factors`: Number of factors
    * `num_features`: Dimensionality
    * `k`: latent factor dimensionality
    * `F`: 0/1 design matrix of shape [num_samples, num_factors]
"""
def linear_regression_model(
        num_factors, num_features, F, k,
        x_bias_loc, x_bias_scale, x_scale_hinges, sample_scales):

    x_bias = ed.Normal(
        loc=tf.fill([num_features], np.float32(x_bias_loc)),
        scale=np.float32(x_bias_scale),
        name="x_bias")


    # w
    # -

    w_global_scale = ed.HalfCauchy(loc=0.0, scale=1.0, name="w_global_scale")
    w_local_scale = ed.HalfCauchy(loc=tf.zeros([num_features]), scale=1.0, name="w_local_scale")

    w = ed.Normal(
        loc=tf.zeros([num_features, k]),
        scale=tf.expand_dims(w_global_scale * w_local_scale, -1),
        name="w")

    v = ed.Normal(
        loc=tf.zeros([k, num_factors]),
        scale=1.0,
        name="v")

    # x
    # -

    x_loc = tf.identity(
        tf.matmul(F, tf.matmul(w, v), transpose_b=True) + x_bias,
        name="x_loc") # [num_samples, num_features]

    x_scale_hinges_diff = tf.square(tf.expand_dims(x_bias, 0) - tf.expand_dims(x_scale_hinges, -1))
    x_scale_hinges_weight_x = tf.exp(-x_scale_hinges_diff) # [scale_spline_degree, num_features]
    x_scale_hinges_weight_x = tf.clip_by_value(x_scale_hinges_weight_x, 1e-12, 1.0)
    x_scale_hinges_weight_x = x_scale_hinges_weight_x / tf.reduce_sum(x_scale_hinges_weight_x, axis=0, keepdims=True)

    x_scale_concentration_c = ed.HalfCauchy(
        loc=tf.zeros([scale_spline_degree]), scale=100.0, name="x_scale_concentration_c")

    x_scale_rate_c = ed.HalfCauchy(
        loc=tf.zeros([scale_spline_degree]), scale=100.0, name="x_scale_rate_c")

    x_scale_concentration_mix = tf.reduce_sum(
        tf.expand_dims(x_scale_concentration_c, -1) * x_scale_hinges_weight_x, axis=0)

    x_scale_rate_mix = tf.reduce_sum(
        tf.expand_dims(x_scale_rate_c, -1) * x_scale_hinges_weight_x, axis=0)

    concentration = x_scale_concentration_mix

    # mode parameterization
    mode = tf.exp(x_scale_rate_mix)
    rate = 1 / ((concentration + 1) * mode)

    # rate parameterization
    # rate = x_scale_rate_mix

    x_scale = ed.InverseGamma(
        concentration=concentration,
        rate=rate,
        name="x_scale")

    x = ed.Normal(
        loc=x_loc - sample_scales,
        scale=x_scale,
        name="x")

    return w_global_scale, w_local_scale, w, v, x_bias, x_scale_concentration_c, x_scale_rate_c, x_scale, x


"""
Variational model for linear regression, to be paired with `linear_regression_model`.
"""
def linear_regression_variational_model(
        qw_global_scale_loc_var,
        qw_local_scale_loc_var, qw_local_scale_scale_var,
        qw_loc_var, qw_scale_var,
        qv_loc_var,
        qx_bias_loc_var, qx_bias_scale_var,
        qx_scale_concentration_c_loc_var,
        qx_scale_rate_c_loc_var,
        qx_scale_loc_var, qx_scale_scale_var,
        qx_loc_var, qx_scale_var,
        use_point_estimates):

    qw_global_scale = ed.Deterministic(
        loc=tf.nn.softplus(qw_global_scale_loc_var),
        name="qw_global_scale")

    qw_local_scale = ed.LogNormal(
        loc=qw_local_scale_loc_var,
        scale=qw_local_scale_scale_var,
        name="qw_local_scale")

    qw = ed.Normal(
        loc=qw_loc_var,
        scale=qw_scale_var,
        name="qw")

    qv = ed.Deterministic(
        loc=qv_loc_var,
        name="qv")

    qx_bias = ed.Normal(
        loc=qx_bias_loc_var,
        scale=qx_bias_scale_var,
        name="qx_bias")

    qx_scale_concentration_c = ed.Deterministic(
        loc=tf.nn.softplus(qx_scale_concentration_c_loc_var),
        name="qx_scale_concentration_c")

    qx_scale_rate_c = ed.Deterministic(
        loc=tf.nn.softplus(qx_scale_rate_c_loc_var),
        name="qx_scale_rate_c")

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

    return qw_global_scale, qw_local_scale, qw, qv, qx_bias, \
        qx_scale_concentration_c, qx_scale_rate_c, qx_scale, qx



"""
Set up a linear regression model for variational inference, returning
"""
def linear_regression_inference(
        init_feed_dict, F, k, x_init, make_likelihood,
        x_bias_mu0, x_bias_sigma0, x_scale_hinges, sample_scales,
        use_point_estimates, sess, niter=30000, elbo_add_term=0.0):

    num_samples = int(F.shape[0])
    num_factors = int(F.shape[1])
    num_features = int(x_init.shape[1])

    log_joint = ed.make_log_joint_fn(
        lambda: linear_regression_model
            (num_factors, num_features, F, k, x_bias_mu0, x_bias_sigma0,
                x_scale_hinges, sample_scales))

    qw_global_scale_loc_var = tf.Variable(-2.0, name="qw_global_scale_loc_var")

    qw_local_scale_loc_var = tf.Variable(
        tf.zeros([num_features]),
        name="qw_local_scale_loc_var")
    qw_local_scale_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_features], -1.0),
        name="qw_local_scale_loc_var"))

    qw_loc_var = tf.Variable(
        tf.zeros([num_features, k]),
        name="qw_loc_var")
    qw_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_features, k], -1.0),
        name="qw_scale_var"))

    qv_loc_var = tf.Variable(
        0.01 * tf.random_normal([k, num_factors]),
        name="qv_loc_var")

    qx_bias_loc_var = tf.Variable(
        tf.reduce_mean(x_init, axis=0),
        name="qx_bias_loc_var")
    qx_bias_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_features], -1.0),
        name="qx_bias_scale_var"))

    qx_scale_concentration_c_loc_var = tf.Variable(
        tf.fill([scale_spline_degree], 0.0),
        name="qx_scale_concentration_c_loc_var")

    qx_scale_rate_c_loc_var = tf.Variable(
        tf.fill([scale_spline_degree], 0.0),
        name="qx_scale_rate_c_loc_var")


    qx_scale_loc_var = tf.Variable(
        tf.fill([num_features], 0.0),
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

    qw_global_scale, qw_local_scale, qw, qv, qx_bias, \
        qx_scale_concentration_c, qx_scale_rate_c, qx_scale, qx = \
        linear_regression_variational_model(
            qw_global_scale_loc_var,
            qw_local_scale_loc_var, qw_local_scale_scale_var,
            qw_loc_var, qw_scale_var,
            qv_loc_var,
            qx_bias_loc_var, qx_bias_scale_var,
            qx_scale_concentration_c_loc_var,
            qx_scale_rate_c_loc_var,
            qx_scale_loc_var, qx_scale_scale_var,
            qx_loc_var, qx_scale_var,
            use_point_estimates)

    log_prior = log_joint(
        w_global_scale=qw_global_scale,
        w_local_scale=qw_local_scale,
        w=qw,
        v=qv,
        x_bias=qx_bias,
        x_scale_concentration_c=qx_scale_concentration_c,
        x_scale_rate_c=qx_scale_rate_c,
        x_scale=qx_scale,
        x=qx)

    variational_log_joint = ed.make_log_joint_fn(
        lambda: linear_regression_variational_model(
            qw_global_scale_loc_var,
            qw_local_scale_loc_var, qw_local_scale_scale_var,
            qw_loc_var, qw_scale_var,
            qv_loc_var,
            qx_bias_loc_var, qx_bias_scale_var,
            qx_scale_concentration_c_loc_var,
            qx_scale_rate_c_loc_var,
            qx_scale_loc_var, qx_scale_scale_var,
            qx_loc_var, qx_scale_var,
            use_point_estimates))

    entropy = variational_log_joint(
        qw_global_scale=qw_global_scale,
        qw_local_scale=qw_local_scale,
        qw=qw,
        qv=qv,
        qx_bias=qx_bias,
        qx_scale=qx_scale,
        qx_scale_concentration_c=qx_scale_concentration_c,
        qx_scale_rate_c=qx_scale_rate_c,
        qx=qx)

    log_likelihood = make_likelihood(qx)

    elbo = log_prior + log_likelihood - entropy + elbo_add_term
    elbo = tf.check_numerics(elbo, "Non-finite ELBO value")

    if sess is None:
        sess = tf.Session()

    train(sess, -elbo, init_feed_dict, niter, 1e-2, decay_rate=.993)

    return (
        sess.run(qx.distribution.mean()),
        sess.run(qw.distribution.mean()),
        sess.run(qw.distribution.stddev()),
        sess.run(qv.distribution.mean()),
        sess.run(qx_bias.distribution.mean()),
        sess.run(qx_scale.distribution.mean()))


def choose_spline_hinges(low, high):
    x_scale_hinges = []
    d = (high - low) / (scale_spline_degree+1)
    for i in range(scale_spline_degree):
        x_scale_hinges.append(low + (i+1)*d)
    return x_scale_hinges


"""
Run variational inference on transcript expression linear regression.
"""
def estimate_transcript_linear_regression(
        init_feed_dict, vars, x_init, F_arr, k,
        sample_scales, use_point_estimates, sess=None, niter=30000):

    F = tf.constant(F_arr, dtype=tf.float32)
    num_features = x_init.shape[1]

    x_init_mean = np.mean(x_init, axis=0)
    x_scale_hinges = choose_spline_hinges(np.min(x_init_mean), np.max(x_init_mean))

    x_bias_mu0 = np.log(1/num_features)
    x_bias_sigma0 = 12.0

    if use_point_estimates:
        make_likelihood = lambda qx: 0.0
    else:
        make_likelihood = lambda qx: rnaseq_approx_likelihood_from_vars(vars, qx)

    return linear_regression_inference(
        init_feed_dict, F, k, x_init, make_likelihood,
        x_bias_mu0, x_bias_sigma0, x_scale_hinges, sample_scales,
        use_point_estimates, sess, niter)


