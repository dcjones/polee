
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

    x_scale_hinges_diff = tf.square(tf.expand_dims(x_bias, 0) - tf.expand_dims(x_scale_hinges, -1))
    x_scale_hinges_weight = tf.exp(-x_scale_hinges_diff) # [scale_spline_degree, num_features]
    x_scale_hinges_weight = x_scale_hinges_weight / tf.reduce_sum(x_scale_hinges_weight, axis=0, keepdims=True)

    # w
    # -

    # horseshoe prior
    tau = ed.HalfCauchy(loc=tf.zeros([scale_spline_degree]), scale=10.0, name="tau")

    w_scale = ed.HalfCauchy(loc=tf.zeros([num_features]), scale=0.1, name="w_scale")

    # [num_features]
    tau_mix = tf.reduce_sum(
        tf.expand_dims(tau, -1) * x_scale_hinges_weight, axis=0)

    w = ed.StudentT(
        df=1.0,
    # w = ed.Normal(
        loc=0.0,
        # scale=tf.expand_dims(tau_mix, -1),
        scale=tf.ones([num_features, num_factors]) * tau[0],
        # scale=tf.ones([1, num_factors]) * tf.expand_dims(w_scale * tau[0], -1),
        # scale=tf.ones([1, num_factors]) * tf.expand_dims(w_scale * tau_mix, -1),
        # scale=tf.fill([num_features, num_factors], 0.06971968),
        name="w")

    # x
    # -

    x_loc = tf.identity(
        tf.matmul(F, w, transpose_b=True) + x_bias,
        name="x_loc")

    # [num_samples, num_features]
    # num_samples = int(F.shape[0])
    # x_loc = tf.ones([num_samples, 1]) * x_bias

    x_scale_c = ed.Normal(
        loc=tf.fill([scale_spline_degree], 0.0), scale=100.0, name="x_scale_c")

    x_scale_loc_mix = tf.reduce_sum(
        tf.expand_dims(x_scale_c, -1) * x_scale_hinges_weight, axis=0)
        # tf.expand_dims([-0.39864218, -0.41476408, -0.5135315, -1.0101233, -1.4462073, -1.5743896, -1.4225029, -1.2755697 ], -1) * x_scale_hinges_weight, axis=0)

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

    return tau, w_scale, w, x_bias, x_scale_c, x_scale, x


"""
Variational model for linear regression, to be paired with `linear_regression_model`.
"""
def linear_regression_variational_model(
        qtau_loc_var, qtau_scale_var,
        qw_scale_loc_var, qw_scale_scale_var,
        qw_loc_var, qw_scale_var,
        qx_bias_loc_var, qx_bias_scale_var,
        qx_scale_c_loc_var, qx_scale_c_scale_var,
        qx_scale_loc_var, qx_scale_scale_var,
        qx_loc_var, qx_scale_var,
        use_point_estimates):

    qtau = ed.Deterministic(
        loc=tf.exp(qtau_loc_var),
        name="qtau")

    qw_scale = ed.LogNormal(
        loc=qw_scale_loc_var,
        scale=qw_scale_scale_var,
        name="qw_scale")

    # qw_scale = ed.Gamma(
    #     concentration=tf.nn.softplus(qw_scale_loc_var),
    #     rate=qw_scale_scale_var,
    #     name="qw_scale")

    # qw_scale = ed.Deterministic(
    #     loc=qw_scale_loc_var,
    #     name="qw_scale")

    qw = ed.Normal(
        loc=qw_loc_var,
        scale=qw_scale_var,
        name="qw")

    qx_bias = ed.Normal(
        loc=qx_bias_loc_var,
        scale=qx_bias_scale_var,
        name="qx_bias")

    qx_scale_c = ed.Deterministic(
        # loc=tf.Print(qx_scale_c_loc_var, [qx_scale_c_loc_var], "qx_scale_c_loc_var"),
        loc=qx_scale_c_loc_var,
        name="qx_scale_c")

    qx_scale = ed.LogNormal(
        loc=qx_scale_loc_var,
        scale=qx_scale_scale_var,
        name="qx_scale")

    # qx_scale = ed.Gamma(
    #     concentration=tf.nn.softplus(qx_scale_loc_var),
    #     rate=qx_scale_scale_var,
    #     name="qx_scale")

    # qx_scale = ed.Deterministic(
    #     loc=qx_scale_loc_var,
    #     name="qx_scale")

    if use_point_estimates:
        qx = ed.Deterministic(loc=qx_loc_var, name="qx")
    else:
        qx = ed.Normal(
            loc=qx_loc_var,
            scale=qx_scale_var,
            name="qx")

    return qtau, qw_scale, qw, qx_bias, qx_scale_c, qx_scale, qx



"""
Set up a linear regression model for variational inference, returning
"""
def linear_regression_inference(
        init_feed_dict, F, x_init, make_likelihood,
        x_bias_mu0, x_bias_sigma0, x_scale_hinges, sample_scales,
        use_point_estimates, sess):

    num_samples = int(F.shape[0])
    num_factors = int(F.shape[1])
    num_features = int(x_init.shape[1])

    log_joint = ed.make_log_joint_fn(
        lambda: linear_regression_model
            (num_factors, num_features, F, x_bias_mu0, x_bias_sigma0, x_scale_hinges))

    qtau_loc_var = tf.Variable(tf.zeros([scale_spline_degree]), name="qtau_loc_var")
    qtau_scale_var = tf.nn.softplus(tf.Variable(tf.zeros([scale_spline_degree]), name="qtau_scale_var"))

    qw_scale_loc_var = tf.Variable(
        tf.zeros([num_features]),
        name="qw_scale_loc_var")
    qw_scale_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_features], -1.0),
        name="qw_scale_loc_var"))

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
        tf.fill([num_features], 2.0),
        name="qx_scale_loc_var")
    qx_scale_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_features], -1.0),
        name="qx_scale_scale_var"))

    qx_loc_var = tf.Variable(
        x_init,
        name="qx_loc_var",
        trainable=not use_point_estimates)

    qx_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_samples, num_features], 1.0),
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

    qtau, qw_scale, qw, qx_bias, qx_scale_c, qx_scale, qx = \
        linear_regression_variational_model(
            qtau_loc_var, qtau_scale_var,
            qw_scale_loc_var, qw_scale_scale_var,
            qw_loc_var, qw_scale_var,
            qx_bias_loc_var, qx_bias_scale_var,
            qx_scale_c_loc_var, qx_scale_c_scale_var,
            qx_scale_loc_var, qx_scale_scale_var,
            qx_loc_var, qx_scale_var,
            use_point_estimates)

    log_prior = log_joint(
        tau=qtau,
        w_scale=qw_scale,
        w=qw,
        x_bias=qx_bias,
        x_scale_c=qx_scale_c,
        x_scale=qx_scale,
        x=qx)

    variational_log_joint = ed.make_log_joint_fn(
        lambda: linear_regression_variational_model(
            qtau_loc_var, qtau_scale_var,
            qw_scale_loc_var, qw_scale_scale_var,
            qw_loc_var, qw_scale_var,
            qx_bias_loc_var, qx_bias_scale_var,
            qx_scale_c_loc_var, qx_scale_c_scale_var,
            qx_scale_loc_var, qx_scale_scale_var,
            qx_loc_var, qx_scale_var,
            use_point_estimates))

    entropy = variational_log_joint(
        qtau=qtau,
        qw_scale=qw_scale,
        qw=qw,
        qx_bias=qx_bias,
        qx_scale=qx_scale,
        qx_scale_c=qx_scale_c,
        qx=qx)

    log_likelihood = make_likelihood(qx)

    scale_penalty = tf.reduce_sum(tfd.Normal(
        loc=sample_scales,
        scale=1e-3).log_prob(tf.log(tf.reduce_sum(tf.exp(qx), axis=1))))

    # x_bias_penalty = tf.reduce_sum(tfd.Normal(
    #     loc=0.0,
    #     scale=1e-3).log_prob(tf.log(tf.reduce_sum(tf.exp(qx_bias)))))

    elbo = log_prior + log_likelihood - entropy + scale_penalty
    # elbo = log_prior + log_likelihood - entropy

    # elbo = log_prior + log_likelihood - entropy

    if sess is None:
        sess = tf.Session()

    # train(sess, -elbo, init_feed_dict, 80000, 1e-4, decay_rate=1.0)
    train(sess, -elbo, init_feed_dict, 20000, 1e-3, decay_rate=0.999)

    # train(sess, -elbo, init_feed_dict, 20000, 1e-3, decay_rate=1.0)
    # train(sess, -elbo, init_feed_dict, 30000, 1e-2, decay_rate=1.0)

    # train(sess, -elbo, init_feed_dict, 20000, 1e-3, decay_rate=1.0,
    #     initialized_vars=set(tf.all_variables()),
    #     var_list=tf.trainable_variables() + [qx_loc_var, qx_scale_var_])

    print("tau")
    print(sess.run(qtau))

    print("x_scale_c")
    print(sess.run(tf.exp(qx_scale_c))[0])

    print("x_bias quantile")
    print(np.quantile(sess.run(qx_bias), [0.0, 0.1, 0.5, 0.9, 1.0]))

    print("x_bias sum")
    print(sess.run(tf.reduce_sum(tf.exp(qx_bias))))

    print("x sums")
    print(sess.run(tf.reduce_sum(tf.exp(qx), axis=1)))

    print("x quantiles")
    for i in range(num_samples):
        print(np.quantile(sess.run(qx[i,:]), [0.0, 0.1, 0.5, 0.9, 1.0]))

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

    make_likelihood = lambda qx: tf.reduce_sum(feature_likelihood.distribution.log_prob(
        tf.log(tf.nn.softmax(qx, axis=1))))
    # make_likelihood = lambda qx: tf.reduce_sum(feature_likelihood.distribution.log_prob(qx))

    F = tf.constant(F_arr, dtype=tf.float32)

    x_init = tf.log(tf.nn.softmax(feature_loc, axis=1))
    # x_init = feature_loc

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
