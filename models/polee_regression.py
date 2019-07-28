
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


scale_polynomial_degree = 3
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

    # w
    # -

    # horseshoe prior
    tau = ed.HalfCauchy(loc=0.0, scale=10.0, name="tau")

    w_scale = ed.HalfCauchy(
        loc=tf.zeros([num_features, num_factors]),
        # loc=tf.zeros([1, 1, 2]),
        scale=0.1,
        name="w_scale")

    # w_scale = ed.HalfNormal(
    #     scale=tf.fill([num_features, num_factors], 1.0),
    #     name="w_scale")

    w = ed.Normal(
        loc=0.0,
        scale=w_scale * tau,
        name="w")

    # w = ed.StudentT(
    #     df=1.0,
    #     loc=tf.fill([num_features, num_factors], 0.0),
    #     scale=tau,
    #     name="w")

    w_mix = ed.Normal(
        loc=tf.fill([2], 0.0),
        scale=10.0,
        name="w_mix")

    # This is like a pseudo spike-and-slab prior. An actualy spike would
    # cause too many problems.

    # w = ed.Mixture(
    #     cat=tfd.Categorical(logits=tf.ones([num_features, num_factors, 1]) * w_mix),
    #     # cat=tfd.Categorical(probs=tf.ones([num_features, num_factors, 1]) * [0.5, 0.5]),
    #     # cat=tfd.Categorical(probs=tf.ones([num_features, num_factors, 1]) * [0.1, 0.9]),
    #     components=[
    #         tfd.Normal(
    #             # df=1.0,
    #             loc=tf.fill([num_features, num_factors], 0.0),
    #             # scale=0.5),
    #             scale=tau),
    #         tfd.Normal(
    #             loc=tf.fill([num_features, num_factors], 0.0),
    #             scale=0.0001)
    #     ],
    #     name="w")

    # x
    # -

    x_bias = ed.Normal(
        loc=tf.fill([num_features], np.float32(x_bias_loc)),
        scale=np.float32(x_bias_scale),
        name="x_bias")

    x_loc = tf.identity(
        tf.matmul(F, w, transpose_b=True) + x_bias,
        name="x_loc")

    x_scale_scale = ed.HalfCauchy(
        loc=tf.fill([scale_spline_degree], 0.0),
        scale=tf.fill([scale_spline_degree], 10.0),
        name="x_scale_scale")

    x_scale_c = ed.Normal(
        loc=tf.zeros([scale_spline_degree]), scale=100.0, name="x_scale_c")

    x_loc_mean = tf.reduce_mean(x_loc, axis=0, keepdims=True)

    # x_scale_loc = \
    #     x_scale_c[0] + \
    #     x_scale_c[1] * x_loc_mean + \
    #     x_scale_c[2] * x_loc_mean**2

    x_scale_hinges_diff = tf.square(x_loc_mean - tf.expand_dims(x_scale_hinges, -1))
    x_scale_hinges_weight = tf.exp(-x_scale_hinges_diff)
    x_scale_hinges_weight = x_scale_hinges_weight / tf.reduce_sum(x_scale_hinges_weight, axis=0, keepdims=True)

    x_scale_loc_mix = tf.reduce_sum(
        tf.expand_dims(x_scale_c, -1) * x_scale_hinges_weight, axis=0)

    x_scale_scale_mix = tf.reduce_sum(
        tf.expand_dims(x_scale_scale, -1) * x_scale_hinges_weight, axis=0)

    x_scale = ed.TransformedDistribution(
        distribution=tfd.Normal(
            loc=x_scale_loc_mix,
            scale=x_scale_scale_mix),
        bijector=tfp.bijectors.Exp(),
        name="x_scale")

    x = ed.Normal(
        loc=x_loc,
        scale=x_scale,
        name="x")

    return tau, w_scale, w_mix, w, x_bias, x_scale_c, x_scale, x_scale_scale, x


"""
Variational model for linear regression, to be paired with `linear_regression_model`.
"""
def linear_regression_variational_model(
        qtau_loc_var, qtau_scale_var,
        qw_scale_loc_var, qw_scale_scale_var,
        qw_mix_loc_var,
        qw_loc_var, qw_scale_var,
        qx_bias_loc_var, qx_bias_scale_var,
        qx_scale_c_loc_var, qx_scale_c_scale_var,
        qx_scale_loc_var, qx_scale_scale_var,
        qx_scale_scale_loc_var, qx_scale_scale_scale_var,
        qx_loc_var, qx_scale_var,
        use_point_estimates):

    # qtau = ed.LogNormal(
    #     loc=qtau_loc_var,
    #     scale=qtau_scale_var,
    #     name="qtau")

    qtau = ed.Deterministic(
        loc=tf.exp(qtau_loc_var),
        name="qtau")

    qw_scale = ed.LogNormal(
        loc=qw_scale_loc_var,
        scale=qw_scale_scale_var,
        name="qw_scale")

    # qw_scale = ed.Deterministic(
    #     loc=tf.exp(qw_scale_loc_var),
    #     name="qw_scale")

    qw_mix = ed.Deterministic(
        loc=qw_mix_loc_var,
        name="qw_mix")

    qw = ed.Normal(
        loc=qw_loc_var,
        scale=qw_scale_var,
        name="qw")

    qx_bias = ed.Normal(
        loc=qx_bias_loc_var,
        scale=qx_bias_scale_var,
        name="qx_bias")

    # qx_scale_c = ed.Normal(
    #     loc=qx_scale_c_loc_var,
    #     scale=qx_scale_c_scale_var,
    #     name="qx_scale_c")

    qx_scale_c = ed.Deterministic(
        loc=qx_scale_c_loc_var,
        name="qx_scale_c")

    qx_scale = ed.LogNormal(
        loc=qx_scale_loc_var,
        scale=qx_scale_scale_var,
        name="qx_scale")

    # qx_scale = ed.Deterministic(
    #     loc=tf.exp(qx_scale_loc_var),
    #     # scale=qx_scale_scale_var,
    #     name="qx_scale")

    # qx_scale_scale = ed.LogNormal(
    #     loc=qx_scale_scale_loc_var,
    #     scale=qx_scale_scale_scale_var,
    #     name="qx_scale_scale")

    qx_scale_scale = ed.Deterministic(
        loc=tf.exp(qx_scale_scale_loc_var),
        name="qx_scale_scale")

    if use_point_estimates:
        qx = ed.Deterministic(loc=qx_loc_var, name="qx")
    else:
        qx = ed.Normal(
            loc=qx_loc_var,
            scale=qx_scale_var,
            name="qx")

    return qtau, qw_scale, qw_mix, qw, qx_bias, qx_scale_c, qx_scale, qx_scale_scale, qx



"""
Set up a linear regression model for variational inference, returning
"""
def linear_regression_inference(
        init_feed_dict, F, param_inits, make_likelihood,
        x_bias_mu0, x_bias_sigma0, x_scale_hinges, use_point_estimates, sess):

    qtau_init, qw_scale_init, qw_mix_init, qw_init, qx_bias_init, \
     qx_scale_scale_init, qx_scale_c_init, qx_scale_init, qx_init = param_inits

    num_samples = int(F.shape[0])
    num_factors = int(F.shape[1])
    num_features = int(qx_init.shape[1])

    log_joint = ed.make_log_joint_fn(
        lambda: linear_regression_model
            (num_factors, num_features, F, x_bias_mu0, x_bias_sigma0, x_scale_hinges))

    qtau_loc_var = tf.Variable(0.0, name="qtau_loc_var")
    qtau_scale_var = tf.nn.softplus(tf.Variable(0.0, name="qtau_scale_var"))

    qw_mix_loc_var = tf.Variable(tf.fill([2], 0.0), name="qw_mix_loc_var")

    qw_scale_loc_var = tf.check_numerics(tf.Variable(
        qw_scale_init,
        name="qx_scale_loc_var"), "qw_scale_loc_var fucked")
    qw_scale_scale_var = tf.check_numerics(tf.nn.softplus(tf.Variable(
        tf.fill([num_features, num_factors], -1.0),
        # tf.fill([1, 1, 2], -1.0),
        name="qx_scale_scale_var")), "qw_scale_scale_var fucked")

    qw_mix_loc_var = tf.Variable(qw_mix_init, name="qw_mix_loc_var")

    qw_loc_var = tf.check_numerics(tf.Variable(
        qw_init,
        name="qx_loc_var"), "qw_loc_var fucked")
    qw_scale_var_ = tf.Variable(
        tf.fill([num_features, num_factors], -1.0),
        name="qx_scale_var")
    qw_scale_var = tf.check_numerics(tf.nn.softplus(qw_scale_var_), "qw_scale_var fucked")

    qx_bias_loc_var = tf.Variable(
        qx_bias_init,
        name="qx_bias_loc_var")
    qx_bias_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_features], -1.0),
        name="qx_bias_scale_var"))

    qxtau_loc_var = tf.Variable(0.0, name="qxtau_loc_var")
    qxtau_scale_var = tf.nn.softplus(tf.Variable(-1.0, name="qxtau_scale_var"))

    qx_scale_c_loc_var = tf.Variable(
        qx_scale_c_init,
        name="qx_scale_c_loc_var")
    qx_scale_c_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([scale_spline_degree], -1.0),
        name="qx_scale_c_scale_var"))

    qx_scale_loc_var = tf.Variable(
        qx_scale_init,
        name="qx_scale_loc_var")
    qx_scale_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_features], -1.0),
        name="qx_scale_scale_var"))

    qx_scale_scale_loc_var = tf.Variable(
        qx_scale_scale_init,
        name="qx_scale_scale_loc_var")
    qx_scale_scale_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([scale_spline_degree], -1.0),
        name="qx_scale_scale_scale_var"))

    qx_loc_var = tf.Variable(
        qx_init,
        name="qx_loc_var",
        # trainable=False)
        trainable=not use_point_estimates)

    qx_scale_var_ = tf.Variable(
        tf.fill([num_samples, num_features], 0.0),
        # trainable=False,
        name="qx_scale_var")

    qx_scale_var = tf.nn.softplus(qx_scale_var_)

    qtau, qw_scale, qw_mix, qw, qx_bias, qx_scale_c, qx_scale, \
        qx_scale_scale, qx = \
        linear_regression_variational_model(
            qtau_loc_var, qtau_scale_var,
            qw_scale_loc_var, qw_scale_scale_var,
            qw_mix_loc_var,
            qw_loc_var, qw_scale_var,
            qx_bias_loc_var, qx_bias_scale_var,
            qx_scale_c_loc_var, qx_scale_c_scale_var,
            qx_scale_loc_var, qx_scale_scale_var,
            qx_scale_scale_loc_var, qx_scale_scale_scale_var,
            qx_loc_var, qx_scale_var,
            use_point_estimates)

    log_prior = log_joint(
        tau=qtau,
        w_scale=qw_scale,
        w_mix=qw_mix,
        w=qw,
        x_bias=qx_bias,
        x_scale_c=qx_scale_c,
        x_scale=qx_scale,
        x_scale_scale=qx_scale_scale,
        x=qx)

    variational_log_joint = ed.make_log_joint_fn(
        lambda: linear_regression_variational_model(
            qtau_loc_var, qtau_scale_var,
            qw_scale_loc_var, qw_scale_scale_var,
            qw_mix_loc_var,
            qw_loc_var, qw_scale_var,
            qx_bias_loc_var, qx_bias_scale_var,
            qx_scale_c_loc_var, qx_scale_c_scale_var,
            qx_scale_loc_var, qx_scale_scale_var,
            qx_scale_scale_loc_var, qx_scale_scale_scale_var,
            qx_loc_var, qx_scale_var,
            use_point_estimates))

    entropy = variational_log_joint(
        qtau=qtau,
        qw_scale=qw_scale,
        qw_mix=qw_mix,
        qw=qw,
        qx_bias=qx_bias,
        qx_scale=qx_scale,
        qx_scale_c=qx_scale_c,
        qx_scale_scale=qx_scale_scale,
        qx=qx)

    log_likelihood = make_likelihood(qx)

    elbo = log_prior + log_likelihood - entropy

    idx = 53091

    # elbo = tf.Print(elbo, [qx[:,idx-1]], "x", summarize=6)
    # elbo = tf.Print(elbo, [qx_scale[idx-1]], "x scale", summarize=6)
    # elbo = tf.Print(elbo, [qw[idx-1,:]], "w", summarize=6)
    # elbo = tf.Print(elbo, [qw_scale[idx-1]], "w scale", summarize=6)

    # elbo = tf.Print(elbo, [qtau], "tau", summarize=6)

    # elbo = tf.Print(elbo, [qw_scale], "w scale", summarize=10)
    # elbo = tf.Print(elbo, [tf.reduce_min(qw_scale), tf.reduce_max(qw_scale)], "w scale extrema", summarize=10)
    # elbo = tf.Print(elbo, [qw], "w", summarize=10)
    # elbo = tf.Print(elbo, [tf.reduce_min(qw), tf.reduce_max(qw)], "w extrema", summarize=10)

    # elbo = tf.Print(elbo, [tf.reduce_mean(qx_loc_var - tf.reduce_mean(qx_loc_var, axis=0), axis=1)], "x scale", summarize=6)
    # elbo = tf.Print(elbo, [tfp.stats.percentile(qx_loc_var - tf.reduce_mean(qx_loc_var, axis=0), 50.0, axis=1)], "x scale", summarize=6)
    # elbo = tf.Print(elbo, [tfp.stats.percentile(qx_loc_var - qx_bias_loc_var, 50.0, axis=1)], "x scale", summarize=6)
    # elbo = tf.Print(elbo, [tf.reduce_mean(qx_loc_var - qx_bias_loc_var, axis=1)], "x scale", summarize=6)


    # tf.summary.scalar("log likelihood", log_likelihood)
    # tf.summary.scalar("log prior", log_prior)
    # tf.summary.scalar("entropy", entropy)
    tf.summary.scalar("elbo", elbo)

    # tf.summary.scalar("qtau mean", qtau.distribution.mean())
    # tf.summary.scalar("qw idx", qw[idx-1,0])
    # tf.summary.scalar("qw_scale idx", qw_scale[idx-1,0])

    tf.summary.histogram("qw", qw)
    # tf.summary.histogram("qw_scale", qw_scale)
    # tf.summary.histogram("qx_scale", tf.exp(qx_log_scale))

    # elbo = tf.Print(elbo, [tf.reduce_min(qx_scale)], "qx_scale min")
    # elbo = tf.Print(elbo, [tf.reduce_min(qw_scale)], "qw_scale min")

    # print(qx_log_scale_scale)
    # print(qx_log_scale_scale.distribution)
    # print(qx_log_scale_scale.distribution.parameters)
    # print(qx_log_scale_scale.distribution.parameters["distribution"].parameters)
    # sys.exit()
    # elbo = tf.Print(elbo, [qx_log_scale_scale], "qx_log_scale_scale")
    # elbo = tf.Print(elbo, [qx_log_scale_scale.distribution.log_prob(qx_log_scale_scale)], "qx_log_scale_scale entropy")

    tf.summary.scalar("c0", qx_scale_c[0])
    tf.summary.scalar("c1", qx_scale_c[1])
    tf.summary.scalar("c2", qx_scale_c[2])

    if sess is None:
        sess = tf.Session()

    train(sess, -elbo, init_feed_dict, 20000, 1e-3, decay_rate=1.0)

    # initialized_vars = set(tf.global_variables())
    # train(sess, -elbo, init_feed_dict, 20000, 1e-3, decay_rate=1.0,
    #     var_list=tf.trainable_variables() + [qx_loc_var, qx_scale_var_],
    #     initialized_vars=initialized_vars)

    # train(sess, -elbo, init_feed_dict, 500000, 1e-5, decay_rate=1.0)
    # train(sess, -elbo, init_feed_dict, 60000, 1e-3, decay_rate=1.0)

    print("c")
    print(sess.run(qx_scale_c))

    print("scale")
    print(sess.run(qx_scale_scale))

    print("qw_scale")
    print(sess.run(qw_scale))

    print("qw_mix")
    print(sess.run(qw_mix))

    print("qtau")
    print(sess.run(qtau))

    print("qx.scale quantiles")
    print(np.quantile(sess.run(qx.distribution.scale), [0.0, 0.1, 0.5, 0.9, 1.0]))

    return (
        sess.run(qx_loc_var),
        sess.run(qw_loc_var),
        sess.run(qw.distribution.scale),
        sess.run(qx_bias.distribution.mean()),
        sess.run(qx_scale.distribution.mean()))



def linear_regression_map_inference(
        init_feed_dict, F, x_init, make_likelihood,
        x_bias_mu0, x_bias_sigma0, x_scale_hinges, use_point_estimates, sess):

    num_samples = int(F.shape[0])
    num_factors = int(F.shape[1])
    num_features = int(x_init.shape[1])

    log_joint = ed.make_log_joint_fn(
        lambda: linear_regression_model
            (num_factors, num_features, F, x_bias_mu0, x_bias_sigma0, x_scale_hinges))

    qtau = tf.nn.softplus(tf.Variable(0.0, name="qtau"))

    qw_scale = tf.nn.softplus(tf.Variable(
        tf.zeros([num_features, num_factors]),
        # tf.zeros([1, 1, 2]),
        # [[[0.1, 10.0]]],
        name="qx_scale"))

    qw_mix = tf.Variable(
        tf.fill([2], 0.0), name="qw_mix")

    qw = tf.Variable(
        tf.zeros([num_features, num_factors]),
        name="qx")

    qx_bias = tf.Variable(
        tf.reduce_mean(x_init, axis=0),
        name="qx_bias")

    qx_scale_scale = tf.nn.softplus(tf.Variable(tf.fill([scale_spline_degree], 0.0), name="qx_scale_scale"))
    qx_scale_c = tf.Variable(tf.zeros([scale_spline_degree]), name="qx_scale_c")
    qx_scale = tf.nn.softplus(tf.Variable(tf.fill([num_features], 1.0), name="qx_scale"))

    qx = tf.Variable(
        x_init,
        name="qx",
        trainable=not use_point_estimates)

    log_prior = log_joint(
        tau=qtau,
        w_scale=qw_scale,
        w_mix=qw_mix,
        w=qw,
        x_bias=qx_bias,
        x_scale_c=qx_scale_c,
        x_scale=qx_scale,
        x_scale_scale=qx_scale_scale,
        x=qx)

    log_likelihood = make_likelihood(qx)

    log_posterior = log_prior + log_likelihood

    if sess is None:
        sess = tf.Session()

    # train(sess, -log_posterior, init_feed_dict, 10000, 1e-3)
    train(sess, -log_posterior, init_feed_dict, 10, 1e-3)

    return sess.run([qtau, qw_scale, qw_mix, qw, qx_bias, qx_scale_scale, qx_scale_c, qx_scale, qx])


"""
Find a good initialization for w by optimizing the point estimate regression.
"""
def find_w_init(F, x_init, sess):

    num_samples = int(F.shape[0])
    num_factors = int(F.shape[1])
    num_features = int(x_init.shape[1])

    x_bias = tf.reduce_mean(x_init, axis=0)

    qw = tf.Variable(
        tf.zeros([num_features, num_factors]),
        name="qw_")

    loss = tf.reduce_sum(tf.square(tf.matmul(F, qw, transpose_b=True) - x_bias))

    train(sess, loss, {}, 2000, 1e-3)

    w_init = sess.run(qw)
    return w_init

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

    # # TODO: testing stuff
    # feature_scale.fill(1e-3)

    feature_likelihood = tfd.Normal(
        loc=feature_loc,
        scale=feature_scale,
        name="feature_likelihood")

    make_likelihood = lambda qx: tf.reduce_sum(feature_likelihood.log_prob(
        tf.log(tf.nn.softmax(qx, axis=1))))

    F = tf.constant(F_arr, dtype=tf.float32)

    x_init = tf.log(tf.nn.softmax(feature_loc, axis=1))

    # choose equally spaced points for piecewise variance function
    x_init_exp = np.exp(feature_loc)
    x_init_mean = np.mean(np.log(x_init_exp / np.sum(x_init_exp, axis=1, keepdims=True)), axis=0)

    # qs = []
    # for i in range(scale_spline_degree):
    #     qs.append((i+1)/(scale_spline_degree+1))
    # x_scale_hinges = np.float32(np.quantile(x_init_mean, qs))

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

    param_inits = \
        linear_regression_map_inference(
            init_feed_dict, F, x_init, make_likelihood,
            x_bias_mu0, x_bias_sigma0, x_scale_hinges, use_point_estimates, sess)

    return linear_regression_inference(
        init_feed_dict, F, param_inits, make_likelihood,
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
