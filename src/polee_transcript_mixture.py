
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import sys
from tensorflow_probability import edward2 as ed
from polee_approx_likelihood import *
from polee_training import *

SIGMA_ALPHA0 = 0.001
SIGMA_BETA0  = 0.001


def estimate_transcript_mixture(
        init_feed_dict, num_samples, n, vars, x0_log,
        num_mix_components, num_pca_components):
    x_bias_loc0 = np.log(1/n)
    x_bias_scale0 = 4.0
    x_bias_prior = tfd.Normal(
        loc=tf.constant(x_bias_loc0, dtype=tf.float32),
        scale=tf.constant(x_bias_scale0, dtype=tf.float32),
        name="x_bias")

    w_prior = tfd.Normal(
        loc=tf.constant(0.0, dtype=tf.float32),
        scale=tf.constant(1.0, dtype=tf.float32),
        name="w_prior")

    # TODO: make this a mixture over some number of components
    # Then we are going to run into the same difficulty as before, with
    # MixtureSameFamily not playing nice with edward.
    z_prior = tfd.Normal(
        loc=tf.constant(0.0, dtype=tf.float32),
        scale=tf.constant(1.0, dtype=tf.float32),
        name="z")

    x_bias = tf.Variable(
        np.mean(x0_log, 0),
        dtype=tf.float32,
        name="x_bias")

    w = tf.Variable(
        tf.random_normal([n, num_pca_components], stddev=0.1),
        dtype=tf.float32,
        name="w")

    z = tf.Variable(
        tf.random_normal([num_samples, num_pca_components], stddev=0.1),
        dtype=tf.float32,
        name="w")

    x_pca = tf.matmul(z, w, transpose_b=True, name="x_pca")

    # TODO: Add some additive error here.
    x = tf.add(x_pca, x_bias, name="x")

    log_likelihood = rnaseq_approx_likelihood_from_vars(vars, x)

    log_prior = \
        tf.reduce_sum(x_bias_prior.log_prob(x_bias)) + \
        tf.reduce_sum(w_prior.log_prob(w)) + \
        tf.reduce_sum(z_prior.log_prob(z))

    log_posterior = log_likelihood + log_prior

    sess = tf.Session()
    train(sess, -log_posterior, init_feed_dict, 500, 2e-2)

    print(sess.run(z))

