
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import sys
from tensorflow_probability import edward2 as ed
from polee_approx_likelihood import *
from polee_training import *


def estimate_transcript_vae_mixture(
        init_feed_dict, num_samples, n, vars, x0_log,
        num_mix_components, num_pca_components):
    pass

    log_prior = 0.0

    # z_mix
    # -----
    z_mix_prior = tfd.Dirichlet(
        concentration=tf.constant(5.0, shape=[num_mix_components]),
        name="z_mix_prior")

    z_mix = tf.Variable(
        tf.zeros([num_mix_components]),
        dtype=tf.float32,
        trainable=False,
        name="z_mix")

    log_prior += tf.reduce_sum(z_mix_prior.log_prob(tf.nn.softmax(z_mix)))

    # z_comp_loc
    # ----------
    z_comp_loc_prior = tfd.Normal(
        loc=tf.constant(0.0, dtype=tf.float32),
        scale=tf.constant(5.0, dtype=tf.float32),
        name="z_comp_loc_prior")

    z_comp_loc = tf.Variable(
        # tf.zeros([num_mix_components, num_pca_components]),
        tf.random_normal([num_mix_components, num_pca_components], stddev=0.1),
        name="z_comp_loc")

    log_prior += tf.reduce_sum(z_comp_loc_prior.log_prob(z_comp_loc))

    # z_comp_scale
    # ------------
    z_comp_scale_prior = HalfCauchy(
        loc=0.0,
        scale=0.01,
        name="z_comp_scale_prior")

    # TODO: allowing this to be trainable completely fucks the whole thing up.
    # Maybe I just need to tinker with the prior. Not sure.
    z_comp_scale = tf.clip_by_value(tf.nn.softplus(tf.Variable(
        # tf.fill([num_mix_components, num_pca_components], -4.0),
        tf.fill([num_mix_components, num_pca_components], 1.0),
        # trainable=False,
        name="z_comp_scale")), 0.01, 100.0)

    z_comp_scale = tf.Print(z_comp_scale, [tf.reduce_min(z_comp_scale), tf.reduce_max(z_comp_scale)], "z_comp_scale span")

    # log_prior += tf.reduce_sum(z_comp_scale_prior.log_prob(z_comp_scale))

    # low dimensional representation
    z_comp_dist = tfd.Normal(
        loc=z_comp_loc,
        scale=z_comp_scale,
        name="z_comp_dist")

    z = tf.Variable(
        tf.random_normal([num_samples, num_pca_components], stddev=0.1),
        name="z")

    z_comp_log_prob = z_comp_dist.log_prob(tf.expand_dims(z, 1))
    z_comp_log_prob += tf.expand_dims(tf.expand_dims(tf.nn.softmax(z_mix), 0), -1)
    z_log_prob = tf.reduce_logsumexp(z_comp_log_prob, 1)

    log_prior += tf.reduce_sum(z_log_prob)

    # x_loc
    # -----

    hidden1 = tf.layers.dense(
        z, 64, activation=tf.nn.relu)

    hidden2 = tf.layers.dense(
        hidden1, 64, activation=tf.nn.relu)

    x_loc = tf.layers.dense(
        hidden2, n, activation=tf.nn.relu)

    # TODO: x error

    # x_scale
    # -------
    x_scale_prior = HalfCauchy(
        loc=0.0,
        scale=0.01,
        name="x_scale_prior")

    x_scale = tf.nn.softplus(tf.Variable(
        tf.fill([n], -4.0), name="x_scale"))

    x_scale = tf.Print(x_scale, [tf.reduce_min(x_scale), tf.reduce_max(x_scale)], "x_scale span")

    # log_prior += tf.reduce_sum(x_scale_prior.log_prob(x_scale))

    # x
    # -
    x_prior = tfd.Normal(
        loc=x_loc,
        scale=x_scale)

    x = tf.Variable(x0_log, name="x")

    log_prior += tf.reduce_sum(x_prior.log_prob(x))

    # likelihood, training

    log_likelihood = rnaseq_approx_likelihood_from_vars(vars, x)
    log_posterior = log_prior + log_likelihood

    sess = tf.Session()
    train(sess, -log_posterior, init_feed_dict, 500, 5e-2)

    print(sess.run(z_comp_log_prob))

    z_comp_log_prob = tf.reduce_sum(z_comp_log_prob, 2)
    component_probs = sess.run(tf.exp(
        z_comp_log_prob - tf.reduce_logsumexp(z_comp_log_prob, 1, keepdims=True)))

    print(sess.run(z))
    print(sess.run(z_comp_loc))

    return component_probs
