
"""
Linear regression with a regression coefficients factored into a low
dimensional space. kind of like doing a regression then PCA on the coefficients,
but in one model.
"""


import numpy as np
from functools import partial
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import sys
# from tensorflow_probability import edward2 as ed
from polee_approx_likelihood import *
from polee_training import *


scale_spline_degree = 10

JDCRoot = tfd.JointDistributionCoroutine.Root

def Independent(dist):
    return tfd.Independent(
        dist,
        reinterpreted_batch_ndims=len(dist.batch_shape))


def SoftplusNormal(loc, scale, name="SoftplusNormal"):
    return tfd.TransformedDistribution(
        distribution=tfd.Normal(
            loc=loc,
            scale=scale),
        bijector=tfp.bijectors.Softplus(),
        name=name)


def mean_variance_model(mean, hinges, concentration_c, mode_c):
    # relative distance from hinges for each value in mean
    diffs = tf.square(tf.expand_dims(mean, 0) - tf.expand_dims(hinges, -1))
    weights = tf.exp(-diffs) # [scale_spline_degree, num_features]
    weights = tf.clip_by_value(weights, 1e-12, 1.0)
    weights = weights / tf.reduce_sum(weights, axis=0, keepdims=True)

    concentration = tf.reduce_sum(
        tf.expand_dims(concentration_c, -1) * weights, axis=0)

    mode = tf.reduce_sum(
        tf.expand_dims(mode_c, -1) * weights, axis=0)

    scale = (concentration + 1) * mode

    return tfd.InverseGamma(
        concentration=concentration,
        scale=scale)


def reduced_rank_regression_inference(
        F, k, x_init, make_likelihood,
        x_bias_loc0, x_bias_scale0, x_scale_hinges, sample_scales,
        use_point_estimates, niter=100000):

    num_samples = int(F.shape[0])
    num_factors = int(F.shape[1])
    num_features = int(x_init.shape[1])

    # decoder network
    # ---------------
    decoder = tf.keras.Sequential()
    decoder.add(tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu))
    decoder.add(tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu))
    decoder.add(tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu))
    decoder.add(tf.keras.layers.Dense(num_features, activation=tf.identity))

    # generative model
    # ----------------
    def model_fn():
        # regression in latent space
        w = yield JDCRoot(Independent(tfd.Normal(
            loc=tf.zeros([num_factors, k]),
            scale=tf.fill([num_factors, k], 10.0))))

        z_scale = yield JDCRoot(Independent(tfd.HalfCauchy(
            loc=tf.zeros([num_factors, k]),
            scale=1.0)))

        z = yield Independent(tfd.Normal(
            loc=tf.matmul(F, w),
            scale=tf.matmul(F, z_scale)))

        x_bias = yield JDCRoot(Independent(tfd.Normal(
            loc=tf.fill([num_features], np.float32(x_bias_loc0)),
            scale=np.float32(x_bias_scale0))))

        # decoded log-expression space
        x_loc = x_bias + decoder(z) - sample_scales
        # x_loc = x_bias

        x_scale_concentration_c = yield JDCRoot(Independent(tfd.HalfCauchy(
            loc=tf.zeros([scale_spline_degree]), scale=1.0)))

        x_scale_mode_c = yield JDCRoot(Independent(tfd.HalfCauchy(
            loc=tf.zeros([scale_spline_degree]), scale=1.0)))

        x_scale = yield Independent(mean_variance_model(
            x_bias, x_scale_hinges, x_scale_concentration_c, x_scale_mode_c))

        # log expression distribution
        x = yield Independent(tfd.StudentT(
            df=1.0,
            loc=x_loc,
            scale=x_scale))
        # tf.print("x: ", x.log_prob(x.sample()))

        if not use_point_estimates:
            rnaseq_reads = yield Independent(make_likelihood(x))


    model = tfd.JointDistributionCoroutine(model_fn)

    # variational model
    # -----------------
    qw_loc_var = tf.Variable(
        tf.random.normal([num_factors, k]))
    qw_softplus_scale_var = tf.Variable(
        tf.fill([num_factors, k], -1.0))

    qz_scale_loc_var = tf.Variable(
        tf.fill([num_factors, k], 0.0))
    qz_scale_softplus_scale_var = tf.Variable(
        tf.fill([num_factors, k], -1.0))

    qz_loc_var = tf.Variable(
        tf.matmul(F, qw_loc_var))
        # tf.random.normal([num_samples, k]))
    qz_softplus_scale_var = tf.Variable(
        tf.fill([num_samples, k], -1.0))

    qx_bias_loc_var = tf.Variable(
        tf.reduce_mean(x_init, axis=0))
    qx_bias_softplus_scale_var = tf.Variable(
        tf.fill([num_features], -1.0))

    qx_scale_concentration_c_loc_var = tf.Variable(
        tf.zeros([scale_spline_degree]))

    qx_scale_mode_c_loc_var = tf.Variable(
        tf.zeros([scale_spline_degree]))

    qx_scale_loc_var = tf.Variable(
        tf.fill([num_features], 0.0))
    qx_scale_softplus_scale_var = tf.Variable(
        tf.fill([num_features], -1.0))

    qx_loc_var = tf.Variable(
        x_init,
        trainable=not use_point_estimates)
    qx_softplus_scale_var = tf.Variable(
        tf.fill([num_samples, num_features], 0.0))


    def variational_model_fn():
        qw = yield JDCRoot(Independent(tfd.Normal(
            loc=qw_loc_var,
            scale=tf.nn.softplus(qw_softplus_scale_var))))

        qz_scale = yield JDCRoot(Independent(SoftplusNormal(
            loc=qz_scale_loc_var,
            scale=tf.nn.softplus(qz_scale_softplus_scale_var))))

        # qz = yield JDCRoot(Independent(tfd.Normal(
        #     loc=qz_loc_var,
        #     scale=tf.nn.softplus(qz_softplus_scale_var))))
        qz = yield JDCRoot(Independent(tfd.Deterministic(
            loc=qz_loc_var)))

        qx_bias = yield JDCRoot(Independent(tfd.Normal(
            loc=qx_bias_loc_var,
            scale=tf.nn.softplus(qx_bias_softplus_scale_var))))

        qx_scale_concentration_c = yield JDCRoot(Independent(tfd.Deterministic(
            loc=tf.nn.softplus(qx_scale_concentration_c_loc_var))))

        qx_scale_mode_c = yield JDCRoot(Independent(tfd.Deterministic(
            loc=tf.nn.softplus(qx_scale_mode_c_loc_var))))

        qx_scale = yield JDCRoot(Independent(SoftplusNormal(
            loc=qx_scale_loc_var,
            scale=tf.nn.softplus(qx_scale_softplus_scale_var))))

        if use_point_estimates:
            qx = yield JDCRoot(Independent(tfd.Deterministic(
                loc=qx_loc_var)))
        else:
            qx = yield JDCRoot(Independent(tfd.Normal(
                loc=qx_loc_var,
                scale=tf.nn.softplus(qx_softplus_scale_var))))

            qrnaseq_reads = yield JDCRoot(tfd.Deterministic(()))

    variational_model = tfd.JointDistributionCoroutine(variational_model_fn)

    # inference
    # ---------

    step_num = tf.Variable(1, trainable=False)

    @tf.function
    def trace_fn(loss, grad, vars):
        if tf.math.mod(step_num, 200) == 0:
            tf.print("[", step_num, "/", niter, "]  loss: ", loss, sep='')
        step_num.assign(step_num + 1)
        return loss

    trace = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=lambda *args: model.log_prob(args),
        surrogate_posterior=variational_model,
        optimizer=tf.optimizers.Adam(learning_rate=1e-3),
        sample_size=1,
        num_steps=niter,
        trace_fn=trace_fn)

    return (
        qw_loc_var.numpy(),
        qz_loc_var.numpy() )


def choose_spline_hinges(low, high):
    x_scale_hinges = []
    d = (high - low) / (scale_spline_degree+1)
    for i in range(scale_spline_degree):
        x_scale_hinges.append(low + (i+1)*d)
    return x_scale_hinges


"""
Run variational inference on transcript expression linear regression.
"""
def estimate_reduced_rank_regression(
        init_feed_dict, vars, x_init, F_arr, k,
        sample_scales, use_point_estimates, sess=None, niter=30000):

    F = tf.constant(F_arr, dtype=tf.float32)
    num_features = x_init.shape[1]

    x_init_mean = np.mean(x_init, axis=0)
    x_scale_hinges = tf.constant(
        choose_spline_hinges(np.min(x_init_mean), np.max(x_init_mean)),
        dtype=tf.float32)

    x_bias_mu0 = np.log(1/num_features)
    x_bias_sigma0 = 12.0

    if use_point_estimates:
        make_likelihood = None
    else:
        make_likelihood = lambda qx: rnaseq_approx_likelihood_from_vars(vars, qx)

    return reduced_rank_regression_inference(
        F, k, x_init, make_likelihood,
        x_bias_mu0, x_bias_sigma0, x_scale_hinges, sample_scales,
        use_point_estimates, niter)


