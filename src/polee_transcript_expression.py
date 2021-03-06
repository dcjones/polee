
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
from polee_approx_likelihood import *
from polee_training import *


def transcript_expression_model(num_samples, n, mu0=None, sigma0=4.0):
    if mu0 is None:
        mu0 = np.log(1.0/n)

    # pooled mean
    x_mu_mu0    = tf.constant(mu0, shape=[n], dtype=tf.float32)
    x_mu_sigma0 = tf.constant(sigma0, shape=[n], dtype=tf.float32)
    x_mu = ed.Normal(
        loc=x_mu_mu0,
        scale=x_mu_sigma0,
        name="x_mu")

    x_sigma_sq_scale0 = tf.constant(0.1, shape=[n], dtype=tf.float32)
    x_sigma_sq = ed.HalfCauchy(
        loc=0.0,
        scale=x_sigma_sq_scale0,
        name="x_sigma_sq")
    x_sigma = tf.sqrt(x_sigma_sq, name="x_sigma")

    # unscaled expression
    x_mu_param = tf.matmul(
        tf.ones([num_samples, 1]), tf.expand_dims(x_mu, 0))
    x = ed.Normal(loc=x_mu_param, scale=x_sigma, name="x")

    return x_mu, x_sigma_sq, x


def transcript_expression_variational_model(
        qx_mu_mu_param, qx_mu_softplus_sigma_param,
        qx_sigma_sq_mu_param, qx_sigma_sq_softplus_sigma_param,
        qx_mu_param, qx_softplus_sigma_param):

    qx_mu = ed.Normal(
        loc=qx_mu_mu_param,
        scale=tf.nn.softplus(qx_mu_softplus_sigma_param),
        name="qx_mu")

    qx_sigma_sq = ed.LogNormal(
        loc=qx_sigma_sq_mu_param,
        scale=tf.nn.softplus(qx_sigma_sq_softplus_sigma_param),
        name="qx_sigma_sq")

    qx = ed.Normal(
        loc=qx_mu_param,
        scale=tf.nn.softplus(qx_softplus_sigma_param),
        name="qx")

    return qx_mu, qx_sigma_sq, qx


def estimate_transcript_expression(init_feed_dict, num_samples, n, vars, x0_log, sess=None):
    log_joint = ed.make_log_joint_fn(
        lambda: transcript_expression_model(num_samples, n))

    qx_mu_mu_param = tf.Variable(
        np.mean(x0_log, 0),
        name="qx_mu_mu_param",
        dtype=tf.float32)
    qx_mu_softplus_sigma_param = tf.Variable(
        tf.fill([n], -1.0),
        name="qx_mu_softplus_sigma_param",
        dtype=tf.float32)

    qx_sigma_sq_mu_param = tf.Variable(
        tf.fill([n], 0.0),
        name="qx_sigma_sq_mu_param",
        dtype=tf.float32)
    qx_sigma_sq_softplus_sigma_param = tf.Variable(
        tf.fill([n], 1.0),
        name="qx_sigma_sq_softplus_sigma_param",
        dtype=tf.float32)

    qx_mu_param = tf.Variable(
        x0_log,
        name="qx_mu_param",
        dtype=tf.float32)
    qx_softplus_sigma_param = tf.Variable(
        tf.fill([num_samples, n], -1.0),
        name="qx_softplus_sigma_param",
        dtype=tf.float32)

    # TODO: try to think of a way to simplify all this stuff
    qx_mu, qx_sigma_sq, qx = transcript_expression_variational_model(
        qx_mu_mu_param, qx_mu_softplus_sigma_param,
        qx_sigma_sq_mu_param, qx_sigma_sq_softplus_sigma_param,
        qx_mu_param, qx_softplus_sigma_param)

    lp = log_joint(
        x_mu=qx_mu,
        x_sigma_sq=qx_sigma_sq,
        x=qx)

    variational_log_joint = ed.make_log_joint_fn(
        lambda: transcript_expression_variational_model(
            qx_mu_mu_param, qx_mu_softplus_sigma_param,
            qx_sigma_sq_mu_param, qx_sigma_sq_softplus_sigma_param,
            qx_mu_param, qx_softplus_sigma_param))

    entropy = -variational_log_joint(
        qx_mu=qx_mu,
        qx_sigma_sq=qx_sigma_sq,
        qx=qx)

    approx_likelihood = rnaseq_approx_likelihood_from_vars(vars, qx)

    elbo = lp + approx_likelihood + entropy

    if sess is None:
        sess = tf.Session()
    train(sess, -elbo, init_feed_dict, 500, 2e-2)

    return sess.run(qx.distribution.loc), sess.run(qx.distribution.scale)
