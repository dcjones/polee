
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sys
from tensorflow_probability import edward2 as ed
from polee_approx_likelihood import *
from polee_training import *

SIGMA_ALPHA0 = 0.001
SIGMA_BETA0  = 0.001


def estimate_transcript_mixture(init_feed_dict, num_samples, n, vars, x0_log, num_components):

    qloc = tf.Variable(
        np.random.uniform(low=0.8, high=1.2, size=(1, num_components, n)) * \
            np.mean(x0_log, 0),
        name="qloc",
        dtype=tf.float32)

    qscale = tf.nn.softplus(tf.Variable(
        tf.fill((1, num_components, n), 1.0),
        name="qscale_softplus"))

    qmix_probs = tf.nn.softmax(
        tf.Variable(
            tf.zeros(num_components),
            name="qmix_probs_logit"))
    qmix_probs = tf.Print(qmix_probs, [qmix_probs], "qmix_probs", summarize=16)

    qx = tf.Variable(x0_log, name="qx")

    log_likelihood = rnaseq_approx_likelihood_from_vars(vars, qx)

    # prior
    mix_probs_prior = tfp.distributions.Dirichlet(
        concentration=tf.ones(num_components),
        name="mix_probs")

    # x_mu_mu0    = tf.constant(np.log(1.0/n), shape=[n], dtype=tf.float32)
    # x_mu_sigma0 = tf.constant(4.0, shape=[n], dtype=tf.float32)
    x_mu_mu0    = tf.constant(np.log(1.0/n), dtype=tf.float32)
    x_mu_sigma0 = tf.constant(4.0, dtype=tf.float32)
    loc_prior = tfp.distributions.Normal(
        loc=x_mu_mu0, scale=x_mu_sigma0, name="loc_prior")

    scale_sq_prior = tfp.distributions.InverseGamma(
        concentration=tf.constant(SIGMA_ALPHA0, dtype=tf.float32),
        rate=tf.constant(SIGMA_BETA0, dtype=tf.float32),
        name="scale_sq_prior")

    mix_dist = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(probs=qmix_probs),
        components_distribution=tfp.distributions.MultivariateNormalDiag(loc=qloc, scale_diag=qscale),
        name="mix")

    tf.summary.histogram("qloc", qloc)
    tf.summary.histogram("qscale", qscale)
    tf.summary.histogram("qmix_probs", qmix_probs)
    tf.summary.histogram("qx", qx)

    log_prior = \
        tf.reduce_sum(mix_probs_prior.log_prob(qmix_probs)) + \
        tf.reduce_sum(loc_prior.log_prob(tf.transpose(qloc))) + \
        tf.reduce_sum(scale_sq_prior.log_prob(tf.square(qscale))) + \
        tf.reduce_sum(mix_dist.log_prob(qx))

    log_posterior = log_prior + log_likelihood

    train(-log_posterior, init_feed_dict, 500, 2e-2)

    # TODO:
    #   Now we want to assign samples to components.
