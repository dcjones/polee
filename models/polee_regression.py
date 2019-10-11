
"""
Bayesian linear regression.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import sys
import polee
from polee_approx_likelihood import *
from polee_gene_expression import *
from polee import *


def linear_regression_inference(
        F, x_init, make_likelihood,
        x_bias_loc0, x_bias_scale0, x_scale_hinges, sample_scales,
        use_point_estimates, kernel_regression_degree, kernel_regression_bandwidth,
        niter):

    num_samples = int(F.shape[0])
    num_factors = int(F.shape[1])
    num_features = int(x_init.shape[1])

    # generative model
    # ----------------

    def model_fn():
        # Horseshoe prior on coefficients
        # Weird parameterization is from tfp.sts.SparseLinearRegression This
        # works because a standard cauchy is a t distribution with df=1, and a
        # t-distribution can be expressed as a gamma-normal compound.

        w_global_scale_variance = yield JDCRoot(Independent(tfd.InverseGamma(
            concentration=0.5, scale=0.5)))
        w_global_scale_noncentered = yield JDCRoot(Independent(tfd.HalfNormal(
            scale=1.0)))
        w_global_scale = w_global_scale_noncentered * tf.sqrt(w_global_scale_variance)

        w_local_scale_variance = yield JDCRoot(Independent(tfd.InverseGamma(
            concentration=tf.fill([num_factors, num_features], 0.5),
            scale=tf.fill([num_factors, num_features], 0.5))))
        w_local_scale_noncentered = yield JDCRoot(Independent(tfd.HalfNormal(
            scale=tf.ones([num_factors, num_features]))))
        w_local_scale = w_local_scale_noncentered * tf.sqrt(w_local_scale_variance)

        w = yield Independent(tfd.Normal(
            loc=tf.zeros([num_factors, num_features]),
            scale=w_global_scale * w_local_scale))

        x_bias = yield JDCRoot(Independent(tfd.Normal(
            loc=tf.fill([num_features], np.float32(x_bias_loc0)),
            scale=np.float32(x_bias_scale0))))

        weights = kernel_regression_weights(
            kernel_regression_bandwidth, x_bias, x_scale_hinges)

        # Kernel regression correction.
        # This can be thought of as encoding the prior that similarly expression
        # genes are probably not all DE in the same direction. This can help
        # in situations where there are large differences in sequencing depth
        # that are highly correlated with factors in the regression. Differences
        # in sequencing depth alter the likelihood function of transcripts with
        # no reads which can otherwise manifest as  bias in the w
        # posterior.

        w_distortion_c = yield Independent(tfd.Normal(
            loc=tf.zeros([num_factors, kernel_regression_degree]),
            scale=1.0))

        w_distortion = tf.matmul(
            tf.expand_dims(w_distortion_c, 1),
            tf.expand_dims(weights, 0)) # [num_factors, num_features]

        # the actual regression part
        x_loc = tf.matmul(F, w + w_distortion) + x_bias

        # Kernel regression mean-variance model
        # Biological variance is InverseGamma distributed with the mode
        # determined by kernel regression against the mean expression.

        x_scale_concentration_c = yield JDCRoot(Independent(tfd.HalfCauchy(
            loc=tf.zeros([kernel_regression_degree]), scale=10.0)))

        x_scale_scale_c = yield JDCRoot(Independent(tfd.HalfCauchy(
            loc=tf.zeros([kernel_regression_degree]), scale=10.0)))

        x_scale = yield Independent(mean_variance_model(
            weights, x_scale_concentration_c, x_scale_scale_c))

        x = yield Independent(tfd.Normal(
            loc=x_loc - sample_scales,
            scale=x_scale))

        if not use_point_estimates:
            # penalty for scale drift
            x_sample_scale = yield Independent(tfd.Normal(
                loc=tf.zeros([num_samples]),
                scale=5e-4))

            rnaseq_reads = yield tfd.Independent(make_likelihood(x))

    model = tfd.JointDistributionCoroutine(model_fn)

    # variational model
    # -----------------

    qw_global_scale_variance_loc_var = tf.Variable(0.0)
    qw_global_scale_variance_softplus_scale_var = tf.Variable(-1.0)

    qw_global_scale_noncentered_loc_var = tf.Variable(0.0)
    qw_global_scale_noncentered_softplus_scale_var = tf.Variable(-1.0)

    qw_local_scale_variance_loc_var = tf.Variable(
        tf.fill([num_factors, num_features], 0.0))
    qw_local_scale_variance_softplus_scale_var = tf.Variable(
        tf.fill([num_factors, num_features], -1.0))

    qw_local_scale_noncentered_loc_var = tf.Variable(
        tf.fill([num_factors, num_features], 0.0))
    qw_local_scale_noncentered_softplus_scale_var = tf.Variable(
        tf.fill([num_factors, num_features], -1.0))

    qw_loc_var = tf.Variable(
        tf.zeros([num_factors, num_features]))
    qw_softplus_scale_var = tf.Variable(
        tf.fill([num_factors, num_features], -2.0))

    qx_bias_loc_var = tf.Variable(
        tf.reduce_mean(x_init, axis=0))
    qx_bias_softplus_scale_var = tf.Variable(
        tf.fill([num_features], -1.0))

    qw_distortion_c_loc_var = tf.Variable(
        tf.zeros([num_factors, kernel_regression_degree]))

    qx_scale_concentration_c_loc_var = tf.Variable(
        tf.fill([kernel_regression_degree], 1.0))

    qx_scale_scale_c_loc_var = tf.Variable(
        tf.fill([kernel_regression_degree], 1.0))

    qx_scale_loc_var = tf.Variable(
        tf.fill([num_features], -0.5))
    qx_scale_softplus_scale_var = tf.Variable(
        tf.fill([num_features], -1.0))

    qx_loc_var = tf.Variable(
        x_init,
        trainable=not use_point_estimates)
    qx_softplus_scale_var = tf.Variable(
        tf.fill([num_samples, num_features], -1.0))

    def variational_model_fn():
        qw_global_scale_variance = yield JDCRoot(Independent(SoftplusNormal(
            loc=qw_global_scale_variance_loc_var,
            scale=tf.nn.softplus(qw_global_scale_variance_softplus_scale_var))))

        qw_global_scale_noncentered = yield JDCRoot(Independent(SoftplusNormal(
            loc=qw_global_scale_noncentered_loc_var,
            scale=tf.nn.softplus(qw_global_scale_noncentered_softplus_scale_var))))

        qw_local_scale_variance = yield JDCRoot(Independent(SoftplusNormal(
            loc=qw_local_scale_variance_loc_var,
            scale=tf.nn.softplus(qw_local_scale_variance_softplus_scale_var))))

        qw_local_scale_noncentered = yield JDCRoot(Independent(SoftplusNormal(
            loc=qw_local_scale_noncentered_loc_var,
            scale=tf.nn.softplus(qw_local_scale_noncentered_softplus_scale_var))))

        qw = yield JDCRoot(Independent(tfd.Normal(
            loc=qw_loc_var,
            scale=tf.nn.softplus(qw_softplus_scale_var))))

        qx_bias = yield JDCRoot(Independent(tfd.Normal(
            loc=qx_bias_loc_var,
            scale=tf.nn.softplus(qx_bias_softplus_scale_var))))

        qw_distortion_c = yield JDCRoot(Independent(tfd.Deterministic(
            loc=qw_distortion_c_loc_var)))

        qx_scale_concentration_c = yield JDCRoot(Independent(tfd.Deterministic(
            loc=tf.nn.softplus(qx_scale_concentration_c_loc_var))))

        qx_scale_scale_c = yield JDCRoot(Independent(tfd.Deterministic(
            loc=tf.nn.softplus(qx_scale_scale_c_loc_var))))

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

            qx_sample_scale = yield Independent(tfd.Deterministic(
                loc=tf.math.log(tf.reduce_sum(tf.math.exp(qx_loc_var), axis=-1))))

            qrnaseq_reads = yield JDCRoot(Independent(tfd.Deterministic(tf.zeros([num_samples]))))

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
        qx_loc_var.numpy(),
        qw_loc_var.numpy(),
        tf.nn.softplus(qw_softplus_scale_var).numpy(),
        qx_bias_loc_var.numpy(),
        tf.nn.softplus(qx_scale_loc_var).numpy())


"""
Run variational inference on transcript expression linear regression.
"""
def estimate_transcript_linear_regression(
        vars, x_init, F_arr,
        sample_scales, use_point_estimates,
        kernel_regression_degree=15, kernel_regression_bandwidth=1.0,
        niter=6000):

    print(F_arr)
    F = tf.constant(F_arr, dtype=tf.float32)
    num_features = x_init.shape[1]

    x_init_mean = np.mean(x_init, axis=0)
    x_scale_hinges = tf.constant(
        choose_knots(np.min(x_init_mean), np.max(x_init_mean), kernel_regression_degree),
        dtype=tf.float32)

    x_bias_mu0 = np.log(1/num_features)
    x_bias_sigma0 = 12.0

    if use_point_estimates:
        make_likelihood = None
    else:
        make_likelihood = lambda qx: rnaseq_approx_likelihood_from_vars(vars, qx)

    return linear_regression_inference(
        F, x_init, make_likelihood,
        x_bias_mu0, x_bias_sigma0, x_scale_hinges, sample_scales,
        use_point_estimates,
        kernel_regression_degree, kernel_regression_bandwidth,
        niter)


"""
Run variational inference on feature log-ratio linear regression.
"""
def estimate_feature_linear_regression(
        feature_loc, feature_scale, feature_sizes,
        F_arr, sample_scales, use_point_estimates,
        kernel_regression_degree=15, kernel_regression_bandwidth=1.0,
        niter=6000):

    num_samples = feature_loc.shape[0]
    num_features = feature_scale.shape[1]

    if use_point_estimates:
        make_likelihood = lambda qx: None
    else:
        make_likelihood = lambda qx: RNASeqFeatureApproxLikelihoodDist(
            feature_loc, feature_scale, feature_sizes, qx)

    F = tf.constant(F_arr, dtype=tf.float32)

    x_init = tf.math.log(tf.nn.softmax(feature_loc, axis=1))

    x_init_mean = np.mean(x_init.numpy(), axis=0)
    x_scale_hinges = choose_knots(np.min(x_init_mean), np.max(x_init_mean), kernel_regression_degree)

    x_bias_mu0 = np.log(1./num_features)
    x_bias_sigma0 = 12.0

    return linear_regression_inference(
        F, x_init, make_likelihood,
        x_bias_mu0, x_bias_sigma0, x_scale_hinges, sample_scales,
        use_point_estimates, kernel_regression_degree, kernel_regression_bandwidth,
        niter)


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
