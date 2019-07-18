
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


scale_polynomial_degree = 4


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
    # tau_scale = 

    tau = ed.HalfCauchy(loc=0.0, scale=10.0, name="tau")

    w_scale = ed.HalfCauchy(
        loc=tf.zeros([num_features, num_factors]),
        # scale=tau, name="w_scale")
        scale=0.1, name="w_scale")

    # w_scale = ed.HalfNormal(
    #     scale=tf.fill([num_features, num_factors], 1.0), name="w_scale")

    w = ed.Normal(
        loc=0.0,
        # scale=tf.maximum(0.1, w_scale * tau),
        scale=w_scale * tau,
        # scale=w_scale,
        # scale=tf.ones([num_features, num_factors]) * tau,
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

    # xtau = ed.HalfCauchy(loc=0.0, scale=1.0, name="tau")

    # TODO: We need a more sophisticated model of biological variance.
    # Other methods work by fitting some nonlinear function to the mean/variance
    # relationship, then shrinking towards the predicted variance. We need to
    # figure out how to do something similar in a bayesian context.

    # xtau = ed.HalfCauchy(loc=0.0, scale=4.0, name="tau")
    # x_scale = ed.HalfCauchy(
    #     loc=tf.fill([num_features], 0.0),
    #     scale=1.0,
    #     # scale=xtau,
    #     name="x_scale")

    x_log_scale_scale = ed.HalfCauchy(
        loc=0.0, scale=1.0, name="x_log_scale_scale")

    x_log_scale_c = ed.Normal(
        loc=tf.zeros([scale_polynomial_degree]), scale=100.0, name="x_log_scale_c")

    x_loc_mean = tf.reduce_mean(x_loc, axis=0)

    x_log_scale = ed.StudentT(
        df=1.0,
        loc=x_log_scale_c[0] + \
            x_log_scale_c[1] * x_loc_mean + \
            x_log_scale_c[2] * x_loc_mean**2,
            # x_log_scale_c[3] * x_loc_mean**3,
        scale=x_log_scale_scale,
        name="x_log_scale")

    # x_log_scale = ed.HalfCauchy(
    #     # df=1.0,
    #     # loc=tf.math.polyval(x_log_scale_c, x_bias),
    #     # loc=x_log_scale_c[0] + x_log_scale_c[1] * x_bias - 3.0,
    #     # loc=x_log_scale_c[0] + x_log_scale_c[1] * tf.reduce_mean(x_loc, axis=0) - 3.0,
    #     # loc=x_log_scale_c[0] + x_log_scale_c[1] * tf.reduce_mean(x_loc, axis=0),
    #     loc=0.0,
    #     scale=tf.exp(x_log_scale_c[0] + x_log_scale_c[1] * tf.reduce_mean(x_loc, axis=0)),
    #     name="x_log_scale")

    # x_scale = ed.InverseGamma(
    #     concentration=tf.fill([num_features], 0.01),
    #     rate=tf.fill([num_features], 0.01), name="x_scale")

    # x = ed.StudentT(
    #     df=10.0,
    x = ed.Normal(
        loc=x_loc,
        # scale=0.50,
        # scale=tf.maximum([0.1], tf.exp(x_log_scale)),
        # scale=1.0,
        # scale=xtau,
        # scale=xtau*x_scale,
        # scale=tf.ones([num_features]) * xtau,
        scale=tf.exp(x_log_scale),
        # scale=x_log_scale,
        # scale=x_scale,
        name="x")

    return tau, w_scale, w, x_bias, x_log_scale_c, x_log_scale, x_log_scale_scale, x
    # return tau, w_scale, w, x_bias, xtau, x_scale, x


"""
Variational model for linear regression, to be paired with `linear_regression_model`.
"""
def linear_regression_variational_model(
        qtau_loc_var, qtau_scale_var,
        qw_scale_loc_var, qw_scale_scale_var,
        qw_loc_var, qw_scale_var,
        qx_bias_loc_var, qx_bias_scale_var,
        # qxtau_loc_var, qxtau_scale_var,
        # qx_scale_loc_var, qx_scale_scale_var,
        qx_log_scale_c_loc_var, qx_log_scale_c_scale_var,
        qx_log_scale_loc_var, qx_log_scale_scale_var,
        qx_log_scale_scale_loc_var, qx_log_scale_scale_scale_var,
        qx_loc_var, qx_scale_var,
        use_point_estimates):

    qtau = ed.LogNormal(
        loc=qtau_loc_var,
        scale=qtau_scale_var,
        name="qtau")

    qw_scale = ed.LogNormal(
        loc=qw_scale_loc_var,
        scale=qw_scale_scale_var,
        name="qw_scale")
    # qw_scale = tf.clip_by_value(qw_scale, 0.001, 10.0)

    # qw_scale = ed.InverseGamma(
    #     concentration=tf.nn.softplus(qw_scale_loc_var),
    #     rate=qw_scale_scale_var,
    #     name="qw_scale")

    qw = ed.Normal(
        loc=qw_loc_var,
        scale=qw_scale_var,
        name="qw")

    qx_bias = ed.Normal(
        loc=qx_bias_loc_var,
        scale=qx_bias_scale_var,
        name="qx_bias")

    # qxtau = ed.LogNormal(
    #     loc=qxtau_loc_var,
    #     scale=qxtau_scale_var,
    #     name="qxtau")

    # qx_scale = ed.LogNormal(
    #     loc=qx_scale_loc_var,
    #     scale=qx_scale_scale_var,
    #     name="qx_scale")

    qx_log_scale_c = ed.Normal(
        loc=qx_log_scale_c_loc_var,
        scale=qx_log_scale_c_scale_var,
        name="qx_log_scale_c")

    qx_log_scale = ed.Normal(
        loc=qx_log_scale_loc_var,
        scale=qx_log_scale_scale_var,
        name="qx_log_scale")

    qx_log_scale_scale = ed.LogNormal(
        loc=qx_log_scale_scale_loc_var,
        scale=qx_log_scale_scale_scale_var,
        name="qx_log_scale_scale")

    # qx_log_scale = ed.LogNormal(
    #     loc=qx_log_scale_loc_var,
    #     scale=qx_log_scale_scale_var,
    #     name="qx_log_scale")

    # qx_log_scale = ed.InverseGamma(
    #     concentration=tf.nn.softplus(qx_log_scale_loc_var),
    #     rate=qx_log_scale_scale_var,
    #     name="qx_log_scale")

    # qx_scale = ed.InverseGamma(
    #     concentration=tf.nn.softplus(qx_scale_loc_var),
    #     rate=qx_scale_scale_var,
    #     name="qx_scale")

    if use_point_estimates:
        qx = ed.Deterministic(loc=qx_loc_var, name="qx")
    else:
        qx = ed.Normal(
            loc=qx_loc_var,
            scale=qx_scale_var,
            name="qx")

    return qtau, qw_scale, qw, qx_bias, qx_log_scale_c, qx_log_scale, qx_log_scale_scale, qx
    # return qtau, qw_scale, qw, qx_bias, qxtau, qx_scale, qx



"""
Set up a linear regression model for variational inference, returning
"""
def linear_regression_inference(
        init_feed_dict, F, x_init, make_likelihood,
        x_bias_mu, x_bias_sigma, use_point_estimates, sess):
    num_samples = int(F.shape[0])
    num_factors = int(F.shape[1])
    num_features = int(x_init.shape[1])

    log_joint = ed.make_log_joint_fn(
        lambda: linear_regression_model
            (num_factors, num_features, F, x_bias_mu, x_bias_sigma))

    qtau_loc_var = tf.Variable(0.0, name="qtau_loc_var")
    qtau_scale_var = tf.nn.softplus(tf.Variable(0.0, name="qtau_scale_var"))

    qw_scale_loc_var = tf.check_numerics(tf.Variable(
        # tf.zeros([num_features, num_factors]),
        tf.fill([num_features, num_factors], -4.0),
        name="qx_scale_loc_var"), "qw_scale_loc_var fucked")
    qw_scale_scale_var = tf.check_numerics(tf.nn.softplus(tf.Variable(
        # tf.zeros([num_features, num_factors]),
        tf.fill([num_features, num_factors], -2.0),
        name="qx_scale_scale_var")), "qw_scale_scale_var fucked")

    qw_loc_var = tf.check_numerics(tf.Variable(
        tf.fill([num_features, num_factors], 0.0),
        # tf.zeros([num_features, num_factors]),
        name="qx_loc_var"), "qw_loc_var fucked")
    qw_scale_var_ = tf.Variable(
        # tf.zeros([num_features, num_factors]),
        tf.fill([num_features, num_factors], -2.0),
        name="qx_scale_var")
    qw_scale_var = tf.check_numerics(tf.nn.softplus(qw_scale_var_), "qw_scale_var fucked")

    qx_bias_loc_var = tf.Variable(
        tf.reduce_mean(x_init, axis=0),
        name="qx_bias_loc_var")
    qx_bias_scale_var = tf.nn.softplus(tf.Variable(
        tf.zeros([num_features]),
        name="qx_bias_scale_var"))

    qxtau_loc_var = tf.Variable(0.0, name="qxtau_loc_var")
    qxtau_scale_var = tf.nn.softplus(tf.Variable(-2.0, name="qxtau_scale_var"))

    # qx_scale_loc_var = tf.Variable(
    #     # tf.zeros([num_features]),
    #     tf.fill([num_features], -1.0),
    #     name="qx_scale_loc_var")

    # qx_scale_scale_var = tf.nn.softplus(tf.Variable(
    #     tf.zeros([num_features]),
    #     # tf.fill([num_features], -2.0),
    #     name="qx_scale_scale_var"))

    qx_log_scale_c_loc_var = tf.Variable(
        tf.zeros([scale_polynomial_degree]),
        name="qx_log_scale_c_loc_var")
    qx_log_scale_c_scale_var = tf.nn.softplus(tf.Variable(
        tf.zeros([scale_polynomial_degree]),
        name="qx_log_scale_c_scale_var"))

    qx_log_scale_loc_var = tf.Variable(
        tf.zeros([num_features]),
        name="qx_log_scale_loc_var")
    qx_log_scale_scale_var = tf.nn.softplus(tf.Variable(
        tf.zeros([num_features]),
        name="qx_log_scale_scale_var"))

    qx_log_scale_scale_loc_var = tf.Variable(
        -1.0,
        name="qx_log_scale_scale_loc_var")
    qx_log_scale_scale_scale_var = tf.nn.softplus(tf.Variable(
        -3.0,
        name="qx_log_scale_scale_scale_var"))

    qx_loc_var = tf.Variable(
        x_init,
        name="qx_loc_var",
        trainable=not use_point_estimates)
    qx_scale_var = tf.nn.softplus(tf.Variable(
        tf.zeros([num_samples, num_features]) ,
        name="qx_scale_var"))

    # qtau, qw_scale, qw, qx_bias, qxtau, qx_scale, qx = \
    qtau, qw_scale, qw, qx_bias, qx_log_scale_c, qx_log_scale, qx_log_scale_scale, qx = \
        linear_regression_variational_model(
            qtau_loc_var, qtau_scale_var,
            qw_scale_loc_var, qw_scale_scale_var,
            qw_loc_var, qw_scale_var,
            qx_bias_loc_var, qx_bias_scale_var,
            # qxtau_loc_var, qxtau_scale_var,
            # qx_scale_loc_var, qx_scale_scale_var,
            qx_log_scale_c_loc_var, qx_log_scale_c_scale_var,
            qx_log_scale_loc_var, qx_log_scale_scale_var,
            qx_log_scale_scale_loc_var, qx_log_scale_scale_scale_var,
            qx_loc_var, qx_scale_var,
            use_point_estimates)

    log_prior = log_joint(
        tau=qtau,
        w_scale=qw_scale,
        w=qw,
        x_bias=qx_bias,
        # xtau=qxtau,
        # x_scale=qx_scale,
        x_log_scale_c=qx_log_scale_c,
        x_log_scale=qx_log_scale,
        x_log_scale_scale=qx_log_scale_scale,
        x=qx)

    variational_log_joint = ed.make_log_joint_fn(
        lambda: linear_regression_variational_model(
            qtau_loc_var, qtau_scale_var,
            qw_scale_loc_var, qw_scale_scale_var,
            qw_loc_var, qw_scale_var,
            qx_bias_loc_var, qx_bias_scale_var,
            # qxtau_loc_var, qxtau_scale_var,
            # qx_scale_loc_var, qx_scale_scale_var,
            qx_log_scale_c_loc_var, qx_log_scale_c_scale_var,
            qx_log_scale_loc_var, qx_log_scale_scale_var,
            qx_log_scale_scale_loc_var, qx_log_scale_scale_scale_var,
            qx_loc_var, qx_scale_var,
            use_point_estimates))

    entropy = variational_log_joint(
        qtau=qtau,
        qw_scale=qw_scale,
        qw=qw,
        qx_bias=qx_bias,
        # qxtau=qxtau,
        # qx_scale=qx_scale,
        qx_log_scale=qx_log_scale,
        qx_log_scale_c=qx_log_scale_c,
        qx_log_scale_scale=qx_log_scale_scale,
        qx=qx)

    log_likelihood = make_likelihood(qx)

    elbo = log_prior + log_likelihood - entropy

    # elbo = tf.Print(elbo, [log_prior, log_likelihood, entropy], "loss parts", summarize=3)

    # idx = 45047
    # elbo = tf.Print(elbo, [qw_loc_var[idx-1], qw_scale_var[idx-1]], "w loc scale of interest")
    # elbo = tf.Print(elbo, [qx_loc_var[:,idx-1]], "x loc of interest", summarize=6)
    # elbo = tf.Print(elbo, [tf.reduce_sum(tf.exp(qx_loc_var), axis=1)], "x scale", summarize=6)

    # qs = tfp.stats.percentile(tf.exp(qx_loc_var), 95.0, axis=1)
    # elbo = tf.Print(elbo, [qs / tf.reduce_mean(qs)], "x scale", summarize=6)
    # elbo = tf.Print(elbo, [qx_log_scale_c_loc_var], "qx_log_scale_c_loc_var", summarize=6)

    # idx = 41408
    # idx = 56843
    # idx = 57930
    # idx = 58706
    # idx = 58283
    # idx = 57973
    # idx = 58715
    # idx = 50369
    idx = 657

    # elbo = tf.Print(elbo, [qx[:,idx-1]], "x", summarize=6)
    # elbo = tf.Print(elbo, [tf.exp(qx_log_scale[idx-1])], "x scale", summarize=6)
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

    # TODO: trying to figure out why we get NaNs in entropy
    # tf.summary.scalar("qtau entropy", tf.reduce_sum(qtau.distribution.log_prob(qtau)))
    # tf.summary.scalar("qw_scale entropy", tf.reduce_sum(qw_scale.distribution.log_prob(qw_scale)))
    # tf.summary.scalar("qw entropy", tf.reduce_sum(qw.distribution.log_prob(qw)))
    # tf.summary.scalar("qx_bias entropy", tf.reduce_sum(qx_bias.distribution.log_prob(qx_bias)))
    # tf.summary.scalar("qxtau entropy", tf.reduce_sum(qxtau.distribution.log_prob(qxtau)))


    # tf.summary.scalar("log likelihood", log_likelihood)
    # tf.summary.scalar("log prior", log_prior)
    # tf.summary.scalar("entropy", entropy)
    tf.summary.scalar("elbo", elbo)

    # tf.summary.scalar("qtau mean", qtau.distribution.mean())
    # tf.summary.scalar("qw idx", qw[idx-1,0])
    # tf.summary.scalar("qw_scale idx", qw_scale[idx-1,0])

    tf.summary.histogram("qx_bias", qx_bias)
    tf.summary.histogram("qw", qw)
    tf.summary.histogram("qx", qx)
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

    # elbo = tf.Print(elbo, [qx_log_scale_c], "qx_log_scale_c", summarize=6)
    tf.summary.scalar("c0", qx_log_scale_c[0])
    tf.summary.scalar("c1", qx_log_scale_c[1])

    if sess is None:
        sess = tf.Session()

    tf.set_random_seed(1)

    # train(sess, -elbo, init_feed_dict, 16000, 1e-1, decay_rate=0.98)

    # train(sess, -elbo, init_feed_dict, 16000, 1e-1, decay_rate=0.99)
    train(sess, -elbo, init_feed_dict, 32000, 1e-3, decay_rate=1.0)

    # train(sess, -elbo, init_feed_dict, 16000, 5e-4)
    # all_vars = set(tf.global_variables())
    # train(sess, -elbo, init_feed_dict, 10000, 1e-2,
    #     var_list=tf.trainable_variables() + [qw_loc_var, qw_scale_var_],
    #     initialized_vars=all_vars)
    # train(sess, -elbo, init_feed_dict, 8000, 1e-2)
    # all_vars = set(tf.global_variables())
    # train(sess, -elbo, init_feed_dict, 8000, 1e-3, initialized_vars=all_vars)

    return (
        sess.run(qx_loc_var),
        sess.run(qw_loc_var),
        sess.run(qw.distribution.scale),
        sess.run(qx_bias.distribution.mean()),
        # sess.run(qx_scale))
        sess.run(tf.exp(qx_log_scale.distribution.mean())))



def linear_regression_map_inference(
        init_feed_dict, F, x_init, make_likelihood,
        x_bias_mu, x_bias_sigma, use_point_estimates, sess):

    num_samples = int(F.shape[0])
    num_factors = int(F.shape[1])
    num_features = int(x_init.shape[1])

    log_joint = ed.make_log_joint_fn(
        lambda: linear_regression_model
            (num_factors, num_features, F, x_bias_mu, x_bias_sigma))

    qtau = tf.nn.softplus(tf.Variable(0.0, name="qtau"))
    w_scale = tf.nn.softplus(tf.Variable(0.0, name="w_scale"))

    qw_scale = tf.nn.softplus(tf.Variable(
        tf.zeros([num_features, num_factors]),
        name="qx_scale"))

    qw_scale_scale_var = tf.nn.softplus(tf.Variable(
        tf.zeros([num_features, num_factors]),
        # tf.fill([num_features, num_factors], -1.0),
        name="qx_scale_scale_var"))

    qw = tf.Variable(
        tf.zeros([num_features, num_factors]),
        name="qx")

    qx_bias = tf.Variable(
        tf.reduce_mean(x_init, axis=0),
        name="qx_bias")

    # qx_log_scale_c = tf.Variable(tf.zeros([scale_polynomial_degree]), name="qx_log_scale_c")
    qx_log_scale_c = tf.Variable([10.0, -1.0], name="qx_log_scale_c")
    qx_log_scale = tf.Variable(tf.zeros([num_features]), name="qx_log_scale")

    # qxtau_loc_var = tf.Variable(0.0, name="qxtau_loc_var")
    # qxtau_scale_var = tf.nn.softplus(tf.Variable(0.0, name="qxtau_scale_var"))

    # qx_scale = tf.nn.softplus(tf.Variable(
    #     tf.zeros([num_features]),
    #     name="qx_scale"))

    # qxtau = tf.nn.softplus(tf.Variable(-1.0, name="qxtau"))

    qx = tf.Variable(
        x_init,
        name="qx",
        trainable=not use_point_estimates)

    log_prior = log_joint(
        tau=qtau,
        w_scale=qw_scale,
        w=qw,
        x_bias=qx_bias,
        # xtau=qxtau,
        # x_scale=qx_scale,
        x_log_scale_c=qx_log_scale_c,
        x_log_scale=qx_log_scale,
        x=qx)

    log_likelihood = make_likelihood(qx)

    log_posterior = log_prior + log_likelihood

    idx = 50369

    # qx_scale = tf.exp(qx_log_scale)

    # log_posterior = tf.Print(log_posterior, [qx[:,idx-1]], "x", summarize=6)
    # log_posterior = tf.Print(log_posterior, [tf.log(tf.nn.softmax(qx))[:,idx-1]], "x normalized", summarize=6)
    # log_posterior = tf.Print(log_posterior, [qx_scale[idx-1]], "x scale", summarize=6)
    # log_posterior = tf.Print(log_posterior, [qw[idx-1,:]], "w", summarize=6)
    # log_posterior = tf.Print(log_posterior, [qw_scale[idx-1]], "w scale", summarize=6)
    # log_posterior = tf.Print(log_posterior, [qtau], "tau", summarize=6)
    # log_posterior = tf.Print(log_posterior, [qxtau], "xtau", summarize=6)
    # log_posterior = tf.Print(log_posterior, [qx_log_scale_c], "qx_log_scale_c", summarize=6)

    if sess is None:
        sess = tf.Session()

    # train(sess, -log_posterior, init_feed_dict, 16000, 1e-3)
    train(sess, -log_posterior, init_feed_dict, 8000, 1e-3)

    return (
        sess.run(qw),
        sess.run(qw_scale),
        sess.run(qx_bias),
        # sess.run(qx_scale))
        sess.run(tf.exp(qx_log_scale)))

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

    feature_likelihood = tfd.Normal(
        loc=feature_loc,
        scale=feature_scale,
        name="feature_likelihood")

    make_likelihood = lambda qx: tf.reduce_sum(feature_likelihood.log_prob(
        tf.log(tf.nn.softmax(qx, axis=1))))

    F = tf.constant(F_arr, dtype=tf.float32)

    x_init = tf.log(tf.nn.softmax(feature_loc, axis=1))

    # TODO: trying different initialization
    # feature_loc_exp = np.exp(feature_loc)
    # x_init = np.log(feature_loc_exp / np.sum(feature_loc_exp))
    # x_init = np.repeat(np.mean(x_init, axis=0, keepdims=True), repeats=x_init.shape[0], axis=0)

    x_bias_mu0 = np.log(1/num_features)
    x_bias_sigma0 = 16.0

    return linear_regression_inference(
        init_feed_dict, F, x_init, make_likelihood,
        x_bias_mu0, x_bias_sigma0, use_point_estimates, sess)

    # return linear_regression_map_inference(
    #     init_feed_dict, F, x_init, make_likelihood,
    #     x_bias_mu0, x_bias_sigma0, use_point_estimates, sess)


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
