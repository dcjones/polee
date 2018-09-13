
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
from polee_approx_likelihood import *


SIGMA_ALPHA0 = 0.001
SIGMA_BETA0  = 0.001

def transcript_expression_model(num_samples, n, vars):
    # pooled mean
    x_mu_mu0    = tf.constant(np.log(1.0/n), shape=[n], dtype=tf.float32)
    x_mu_sigma0 = tf.constant(4.0, shape=[n], dtype=tf.float32)
    x_mu = ed.Normal(
        loc=x_mu_mu0,
        scale=x_mu_sigma0,
        name="x_mu")

    # pooled variance
    x_sigma_alpha0 = tf.constant(SIGMA_ALPHA0, shape=[n], dtype=tf.float32)
    x_sigma_beta0 = tf.constant(SIGMA_BETA0, shape=[n], dtype=tf.float32)
    x_sigma_sq = ed.InverseGamma(
        concentration=x_sigma_alpha0,
        rate=x_sigma_beta0,
        name="x_sigma_sq")
    x_sigma = tf.sqrt(x_sigma_sq, name="x_sigma")

    # unscaled expression
    x = ed.Normal(loc=x_mu, scale=x_sigma, name="x")

    # likelihood
    approx_likelihood = rnaseq_approx_likelihood_from_vars(vars, x)

    # TODO: what do I return here? Every random variables?
    return x


def estimate_transcript_expression(num_samples, n, vars):
    transcript_expression_model(num_samples, n, vars)
    # TODO:

