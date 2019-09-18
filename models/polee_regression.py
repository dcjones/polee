
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

from tensorflow.python import debug as tf_debug


kernel_regression_degree = 10
kernel_regression_bandwidth = 1.0


def gaussian_kernel(u, bandwidth):
    return tf.exp(-tf.square(u))


def SoftplusNormal(loc, scale, name="SoftplusNormal"):
    return ed.TransformedDistribution(
        distribution=tfd.Normal(
            loc=loc,
            scale=scale),
        bijector=tfp.bijectors.Softplus(),
        name=name)


"""
Define model for linear regression.
    * `num_factors`: Number of factors
    * `num_features`: Dimensionality
    * `F`: 0/1 design matrix of shape [num_samples, num_factors]
"""
def linear_regression_model(
        num_factors, num_features, F,
        x_bias_loc, x_bias_scale, x_scale_hinges, sample_scales):

    num_samples = len(sample_scales)

    x_bias = ed.Normal(
        loc=tf.fill([num_features], np.float32(x_bias_loc)),
        scale=np.float32(x_bias_scale),
        name="x_bias")

    hinge_weights = gaussian_kernel(
        tf.expand_dims(x_bias, 0) - tf.expand_dims(x_scale_hinges, -1),
        kernel_regression_bandwidth)
    hinge_weights = tf.clip_by_value(hinge_weights, 1e-12, 1.0)
    hinge_weights = hinge_weights / tf.reduce_sum(hinge_weights, axis=0, keepdims=True)

    # w
    # -

    # Different horseshoe parameterization used in tfp.sts.SparseLinearRegression
    # This works because a standard cauchy is a t distribution with df=1, and
    # a t-distribution can be expressed as a gamma-normal compound.

    w_global_scale_variance = ed.InverseGamma(
        concentration=0.5, scale=0.5, name="w_global_scale_variance")
    w_global_scale_noncentered = ed.HalfNormal(
        scale=1.0, name="w_global_scale_noncentered")
    w_global_scale = w_global_scale_noncentered * tf.sqrt(w_global_scale_variance)

    w_local_scale_variance = ed.InverseGamma(
        concentration=tf.fill([num_features], 0.5),
        scale=tf.fill([num_features], 0.5),
        name="w_local_scale_variance")
    w_local_scale_noncentered = ed.HalfNormal(
        scale=tf.ones([num_features]),
        name="w_local_scale_noncentered")
    w_local_scale = w_local_scale_noncentered * tf.sqrt(w_local_scale_variance)

    w = ed.Normal(
        loc=tf.zeros([num_features, num_factors]),
        scale=tf.expand_dims(w_global_scale * w_local_scale, -1),
        name="w")

    # x
    # -

    w_distortion_c = ed.Normal(
        loc=tf.zeros([num_factors, kernel_regression_degree]),
        scale=1.0,
        name="w_distortion_c")

    w_distortion = tf.expand_dims(tf.squeeze(tf.matmul(
        tf.expand_dims(w_distortion_c, 1),
        tf.expand_dims(hinge_weights, 0))), -1)

    x_loc = tf.identity(
        tf.matmul(F, w + w_distortion, transpose_b=True) + x_bias,
        name="x_loc") # [num_samples, num_features]

    x_scale_concentration_c = ed.HalfCauchy(
        loc=tf.zeros([kernel_regression_degree]), scale=100.0, name="x_scale_concentration_c")

    x_scale_rate_c = ed.HalfCauchy(
        loc=tf.zeros([kernel_regression_degree]), scale=100.0, name="x_scale_rate_c")

    x_scale_concentration_mix = tf.squeeze(tf.matmul(
        hinge_weights,
        tf.expand_dims(x_scale_concentration_c, -1),
        transpose_a=True))

    x_scale_rate_mix = tf.squeeze(tf.matmul(
        hinge_weights,
        tf.expand_dims(x_scale_rate_c, -1),
        transpose_a=True))

    concentration = x_scale_concentration_mix

    # mode parameterization
    mode = tf.exp(x_scale_rate_mix)
    rate = 1 / ((concentration + 1) * mode)

    x_scale = ed.InverseGamma(
        concentration=concentration,
        rate=rate,
        name="x_scale")

    # x_distortion_c_ = tf.log(tf.nn.softmax(x_distortion_c, axis=1)) - np.log(1/scale_spline_degree)

    # x_distortion = tf.squeeze(tf.matmul(
    #     tf.expand_dims(x_distortion_c, 1),
    #     tf.expand_dims(x_scale_hinges_weight_x, 0)))

    # x_distortion = evalpoly(x_distortion_c, tf.expand_dims(x_bias, 0))

    # x_distortion -= tf.reduce_mean(x_distortion, axis=1, keepdims=True)
    # x_distortion = tf.log(tf.nn.softmax(x_distortion, axis=1) * num_features)
    # x_distortion = tf.check_numerics(x_distortion, "x_distortion error")

    # center distortion so it doesn't cause scale changes

    x = ed.Normal(
        # loc=x_loc - sample_scales + x_distortion,
        loc=x_loc - sample_scales,
        scale=x_scale,
        name="x")

    return w_global_scale_variance, w_global_scale_noncentered, \
        w_local_scale_variance, w_local_scale_noncentered, \
        w, x_bias, x_scale_concentration_c, x_scale_rate_c, x_scale, w_distortion_c, x


"""
Variational model for linear regression, to be paired with `linear_regression_model`.
"""
def linear_regression_variational_model(
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
        qx_loc_var, qx_scale_var,
        use_point_estimates):

    # qw_global_scale = ed.Deterministic(
    #     loc=tf.nn.softplus(qw_global_scale_loc_var),
    #     name="qw_global_scale")

    # qw_global_scale = ed.LogNormal(
    #     loc=qw_global_scale_loc_var,
    #     scale=qw_global_scale_scale_var,
    #     name="qw_global_scale")

    # qw_local_scale_loc = ed.InverseGamma(
    #     loc=tf.nn.softplus(qw_local_scale_loc_loc_var),
    #     scale=tf.nn.softplus(qw_local_scale_loc_scale_var),
    #     name="qw_local_scale_loc")

    # qw_local_scale = ed.InverseGamma(
    #     loc=tf.nn.softplus(qw_local_scale_loc_var),
    #     scale=qw_local_scale_scale_var,
    #     name="qw_local_scale")

    qw_global_scale_variance = SoftplusNormal(
        loc=qw_global_scale_variance_loc_var,
        scale=qw_global_scale_variance_scale_var,
        name="qw_global_scale_variance")

    qw_global_scale_noncentered = SoftplusNormal(
        loc=qw_global_scale_noncentered_loc_var,
        scale=qw_global_scale_noncentered_scale_var,
        name="qw_global_scale_noncentered")

    qw_local_scale_variance = SoftplusNormal(
        loc=qw_local_scale_variance_loc_var,
        scale=qw_local_scale_variance_scale_var,
        name="qw_local_scale_variance")

    qw_local_scale_noncentered = SoftplusNormal(
        loc=qw_local_scale_noncentered_loc_var,
        scale=qw_local_scale_noncentered_scale_var,
        name="qw_local_scale_noncentered")

    # qw_local_scale_loc = ed.Deterministic(
    #     loc=tf.exp(qw_local_scale_loc_loc_var),
    #     name="qw_local_scale_loc")

    # qw_local_scale = ed.Deterministic(
    #     loc=tf.exp(qw_local_scale_loc_var),
    #     name="qw_local_scale")

    qw = ed.Normal(
        loc=qw_loc_var,
        scale=qw_scale_var,
        name="qw")
    # qw = ed.Deterministic(
    #     loc=tf.zeros(qw_loc_var.shape),
    #     name="qw")

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

    # qx_scale_concentration_c = ed.Deterministic(
    #     loc=qx_scale_concentration_c_loc_var,
    #     name="qx_scale_concentration_c")

    # qx_scale_rate_c = ed.Deterministic(
    #     loc=qx_scale_rate_c_loc_var,
    #     name="qx_scale_rate_c")

    qx_scale = SoftplusNormal(
        loc=qx_scale_loc_var,
        scale=qx_scale_scale_var,
        name="qx_scale")

    # qx_scale = ed.Gamma(
    #     concentration=qx_scale_loc_var,
    #     rate=qx_scale_scale_var,
    #     name="qx_scale")

    qw_distortion_c = ed.Deterministic(
        loc=qw_distortion_c_loc_var,
        name="qw_distortion_c")

    if use_point_estimates:
        qx = ed.Deterministic(loc=qx_loc_var, name="qx")
    else:
        qx = ed.Normal(
            loc=qx_loc_var,
            scale=qx_scale_var,
            name="qx")

    return qw_global_scale_variance, qw_global_scale_noncentered, \
        qw_local_scale_variance, qw_local_scale_noncentered, qw, qx_bias, \
        qx_scale_concentration_c, qx_scale_rate_c, qx_scale, qw_distortion_c, qx


def softminus(x):
    return tf.log(tf.exp(x) - 1.0)


"""
Set up a linear regression model for variational inference, returning
"""
def linear_regression_inference(
        init_feed_dict, F, x_init, make_likelihood,
        x_bias_mu0, x_bias_sigma0, x_scale_hinges, sample_scales,
        use_point_estimates, sess, niter=6000, elbo_add_term=0.0):

    num_samples = int(F.shape[0])
    num_factors = int(F.shape[1])
    num_features = int(x_init.shape[1])

    log_joint = ed.make_log_joint_fn(
        lambda: linear_regression_model
            (num_factors, num_features, F, x_bias_mu0, x_bias_sigma0, x_scale_hinges, sample_scales))

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
    # qw_local_scale_noncentered_loc_var = tf.Print(
    #     qw_local_scale_noncentered_loc_var,
    #     [tf.reduce_min(qw_local_scale_noncentered_loc_var),
    #     tf.reduce_max(qw_local_scale_noncentered_loc_var) ],
    #     "qw_local_scale_noncentered_loc_var span")
    # qw_local_scale_noncentered_scale_var = tf.Print(
    #     qw_local_scale_noncentered_scale_var,
    #     [tf.reduce_min(qw_local_scale_noncentered_scale_var),
    #     tf.reduce_max(qw_local_scale_noncentered_scale_var) ],
    #     "qw_local_scale_noncentered_scale_var span")

    qw_loc_var = tf.Variable(
        tf.zeros([num_features, num_factors]),
        name="qw_loc_var")
    # qw_loc_var = tf.Print(qw_loc_var, [tfp.stats.percentile(qw_loc_var, [45, 50, 55])], "w median")
    qw_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_features, num_factors], -2.0),
        name="qw_scale_var"))

    qx_bias_loc_var = tf.Variable(
        tf.reduce_mean(x_init, axis=0),
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

    qx_loc_var = tf.Variable(
        x_init,
        name="qx_loc_var",
        trainable=not use_point_estimates)

    qx_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_samples, num_features], -1.0),
        name="qx_scale_var"))

    # qx_loc_var = tf.Variable(
    #     x_init,
    #     name="qx_loc_var",
    #     trainable=False)

    # qx_scale_var_ = tf.Variable(
    #     tf.fill([num_samples, num_features], -5.0),
    #     name="qx_scale_var",
    #     trainable=False)
    # qx_scale_var = tf.nn.softplus(qx_scale_var_)

    qw_global_scale_variance, qw_global_scale_noncentered, \
        qw_local_scale_variance, qw_local_scale_noncentered, qw, qx_bias, \
        qx_scale_concentration_c, qx_scale_rate_c, qx_scale, \
        qw_distortion_c, qx = \
        linear_regression_variational_model(
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
            qx_loc_var,                          qx_scale_var,
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
        x=qx)

    variational_log_joint = ed.make_log_joint_fn(
        lambda: linear_regression_variational_model(
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
            qx_loc_var,                          qx_scale_var,
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
        qx=qx)

    log_likelihood = make_likelihood(qx)
    log_likelihood = tf.Print(log_likelihood, [tf.reduce_sum(tf.exp(qx), axis=1)], "scales", summarize=6)
    log_likelihood = tf.Print(log_likelihood, [tf.reduce_sum(tf.exp(qx_bias))], "bias scale", summarize=6)

    scale_penalty = tf.reduce_sum(tfd.Normal(loc=0.0, scale=1e-3).log_prob(
        tf.log(tf.reduce_sum(tf.exp(qx), axis=1))))
    # qx_loc = tf.matmul(F, qw, transpose_b=True) + qx_bias
    # scale_penalty += tf.reduce_sum(tfd.Normal(loc=0.0, scale=1e-3).log_prob(
    #     tf.log(tf.reduce_sum(tf.exp(qx_loc), axis=1))))

    # scale_penalty = tf.reduce_sum(tfd.StudentT(df=10.0, loc=-25.0, scale=tf.fill([1, num_features], 1e-2)).log_prob(
    #     qx))

    elbo = log_prior + log_likelihood - entropy + elbo_add_term # + scale_penalty

    # elbo = tf.Print(elbo, [log_prior], "log_prior")
    # elbo = tf.Print(elbo, [log_likelihood], "log_likelihood")
    # elbo = tf.Print(elbo, [entropy], "entropy")

    # elbo = tf.Print(elbo, [tf.reduce_min(qw_global_scale_variance), tf.reduce_max(qw_global_scale_variance)], "qw_global_scale_variance span")
    # elbo = tf.Print(elbo, [tf.reduce_min(qw_global_scale_noncentered), tf.reduce_max(qw_global_scale_noncentered)], "qw_global_scale_noncentered span")
    # elbo = tf.Print(elbo, [tf.reduce_min(qw_local_scale_variance), tf.reduce_max(qw_local_scale_variance)], "qw_local_scale_variance span")
    # elbo = tf.Print(elbo, [tf.reduce_min(qw_local_scale_noncentered), tf.reduce_max(qw_local_scale_noncentered)], "qw_local_scale_noncentered span")

    # elbo = tf.Print(elbo, [
    #     tf.reduce_min(tfd.HalfNormal(scale=1.0).log_prob(qw_local_scale_noncentered))],
    #     "qw_local_scale_noncentered log_prob min")

    # elbo = tf.Print(elbo, [
    #     tf.reduce_sum(qw_local_scale_noncentered.distribution.log_prob(qw_local_scale_noncentered))],
    #     "qw_local_scale_noncentered entropy")

    # elbo = tf.Print(elbo, [
    #     qw_global_scale_noncentered.distribution.log_prob(qw_global_scale_noncentered)],
    #     "qw_global_scale_noncentered entropy")

    elbo = tf.check_numerics(elbo, "Non-finite ELBO value")

    if sess is None:
        sess = tf.Session()

    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    # train(sess, -elbo, init_feed_dict, 8000, 1e-2, decay_rate=0.993)

    train(sess, -elbo, init_feed_dict, 5000, 1e-3, decay_rate=1.0)
    # train(sess, -elbo, init_feed_dict, 10000, 1e-3, decay_rate=1.0)

    # train(sess, -elbo, init_feed_dict, niter, 1e-2, decay_rate=.993)
    # train(sess, -elbo, init_feed_dict, 8000, 1e-2, decay_rate=.993)
    # train(sess, -elbo, init_feed_dict, niter, 1e-3, decay_rate=1.0)

    # train(
    #     sess, -elbo, init_feed_dict, 30000, 1e-3,
    #     decay_boundaries=[10000, 20000],
    #     decay_values=[1e-2, 1e-3, 1e-4])

    # Print mean values for everything so I can better initialize these
    # print("mean values")
    # print(sess.run([
    #     qw_global_scale_loc_var,
    #     tf.reduce_mean(qw_local_scale_loc_var),
    #     tf.reduce_mean(softminus(qw_local_scale_scale_var)),
    #     tf.reduce_mean(qw_loc_var),
    #     tf.reduce_mean(softminus(qw_scale_var)),
    #     tf.reduce_mean(qx_bias_loc_var),
    #     tf.reduce_mean(softminus(qx_bias_scale_var)),
    #     tf.reduce_mean(qx_scale_concentration_c_loc_var),
    #     tf.reduce_mean(qx_scale_rate_c_loc_var),
    #     tf.reduce_mean(qx_scale_loc_var),
    #     tf.reduce_mean(softminus(qx_scale_scale_var)),
    #     tf.reduce_mean(softminus(qx_scale_var)) ] ))

    print("qw_distortion_c")
    print(sess.run(qw_distortion_c))

    print("qx_scale_concentration_c")
    print(sess.run(qx_scale_concentration_c))

    print("qx_scale_rate_c")
    print(sess.run(qx_scale_rate_c))

    print("qw_global_scale")
    print(sess.run(qw_global_scale_variance_loc_var))
    print(sess.run(qw_global_scale_variance_scale_var))

    print(sess.run(qw_global_scale_noncentered_loc_var))
    print(sess.run(qw_global_scale_noncentered_scale_var))

    print("qw_local_scale quantiles")
    print(np.quantile(sess.run(qw_local_scale_variance_loc_var), [0.0, 0.1, 0.5, 0.9, 1.0]))
    print(np.quantile(sess.run(qw_local_scale_variance_scale_var), [0.0, 0.1, 0.5, 0.9, 1.0]))

    print(np.quantile(sess.run(qw_local_scale_noncentered_loc_var), [0.0, 0.1, 0.5, 0.9, 1.0]))
    print(np.quantile(sess.run(qw_local_scale_noncentered_scale_var), [0.0, 0.1, 0.5, 0.9, 1.0]))

    print("qx_scale quantiles")
    print(np.quantile(sess.run(qx_scale), [0.0, 0.1, 0.5, 0.9, 1.0]))
    print(np.quantile(sess.run(qx_scale_loc_var), [0.0, 0.1, 0.5, 0.9, 1.0]))
    print(np.quantile(sess.run(qx_scale_scale_var), [0.0, 0.1, 0.5, 0.9, 1.0]))

    print("qx")
    print(np.quantile(sess.run(qx_loc_var), [0.0, 0.1, 0.5, 0.9, 1.0]))
    print(np.quantile(sess.run(qx_scale_var), [0.0, 0.1, 0.5, 0.9, 1.0]))

    print("qw")
    print(np.quantile(sess.run(qw_loc_var), [0.0, 0.1, 0.5, 0.9, 1.0]))
    print(np.quantile(sess.run(qw_scale_var), [0.0, 0.1, 0.5, 0.9, 1.0]))

    return (
        sess.run(qx.distribution.mean()),
        sess.run(qw.distribution.mean()),
        sess.run(qw.distribution.stddev()),
        sess.run(qx_bias.distribution.mean()),
        # sess.run(qx_scale.distribution.mean()))
        sess.run(qx_scale))



def choose_knots(low, high):
    x_scale_hinges = []
    d = (high - low) / (kernel_regression_degree+1)
    for i in range(kernel_regression_degree):
        x_scale_hinges.append(low + (i+1)*d)
    return x_scale_hinges


def choose_knots_from_quants(x_mean):
    qs = []
    d = 1/(kernel_regression_degree+1)
    for i in range(kernel_regression_degree):
        qs.append((i+1)*d)
    return np.quantile(x_mean, qs)

"""
Run variational inference on transcript expression linear regression.
"""
def estimate_transcript_linear_regression(
        init_feed_dict, vars, x_init, F_arr,
        sample_scales, use_point_estimates, sess=None, niter=6000):

    F = tf.constant(F_arr, dtype=tf.float32)
    num_features = x_init.shape[1]

    x_init_mean = np.mean(x_init, axis=0)
    x_scale_hinges = choose_knots(np.min(x_init_mean), np.max(x_init_mean))

    x_bias_mu0 = np.log(1/num_features)
    x_bias_sigma0 = 12.0

    if use_point_estimates:
        make_likelihood = lambda qx: 0.0
    else:
        make_likelihood = lambda qx: rnaseq_approx_likelihood_from_vars(vars, qx)

    return linear_regression_inference(
        init_feed_dict, F, x_init, make_likelihood,
        x_bias_mu0, x_bias_sigma0, x_scale_hinges, sample_scales,
        use_point_estimates, sess, niter)


"""
Run variational inference on feature log-ratio linear regression.
"""
def estimate_feature_linear_regression(
        init_feed_dict, feature_loc, feature_scale, x0_log, F_arr, sample_scales,
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

    if use_point_estimates:
        make_likelihood = lambda qx: 0.0
    else:
        make_likelihood = lambda qx: tf.reduce_sum(feature_likelihood.distribution.log_prob(
            tf.log(tf.nn.softmax(qx, axis=1))))

    F = tf.constant(F_arr, dtype=tf.float32)

    x_init = tf.log(tf.nn.softmax(feature_loc, axis=1))

    if sess is None:
        sess = tf.Session()

    x_init_mean = np.mean(sess.run(x_init), axis=0)
    x_scale_hinges = choose_knots(np.min(x_init_mean), np.max(x_init_mean))

    x_bias_mu0 = np.log(1./num_features)
    x_bias_sigma0 = 12.0

    # x_bias_mu0 = np.log(1e-4/num_features)
    # x_bias_sigma0 = 2.0

    # different initialization
    # x_init = np.tile(x_init_mean, (num_samples, 1))


    return linear_regression_inference(
        init_feed_dict, F, x_init, make_likelihood,
        x_bias_mu0, x_bias_sigma0, x_scale_hinges, sample_scales, use_point_estimates, sess)


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
    w_bias_sigma0 = 20.0

    return linear_regression_inference(
        init_feed_dict, X, x_init, make_likelihood,
        w_mu0, w_sigma0, w_bias_mu0, w_bias_sigma0)
