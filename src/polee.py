
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from polee_gene_expression import *
from polee_splicing import *


# Handly alias
JDCRoot = tfd.JointDistributionCoroutine.Root

"""
Independent distribution that reinterprets all batch dimensions.
"""
def Independent(dist):
    return tfd.Independent(
        dist,
        reinterpreted_batch_ndims=len(dist.batch_shape))


"""
Useful for variational approximations. A bit more numerically stable than
LogNormal approximations.
"""
def SoftplusNormal(loc, scale, name="SoftplusNormal"):
    return tfd.TransformedDistribution(
        distribution=tfd.Normal(
            loc=loc,
            scale=scale),
        bijector=tfp.bijectors.Softplus(),
        name=name)


def gaussian_kernel(u, bandwidth):
    return tf.exp(-tf.square(u/bandwidth))


def kernel_regression_weights(bandwidth, mean, hinges):
    # [num_monte_carlo_samples, num_hinges, num_features]
    diffs = tf.expand_dims(mean, -2) - tf.expand_dims(hinges, -1)
    weights = gaussian_kernel(diffs, bandwidth)
    weights = tf.clip_by_value(weights, 1e-10, 1.0)
    weights = weights / tf.reduce_sum(weights, axis=-2, keepdims=True)
    return weights


"""
Modeling mean-variance relationship using kernel regression.
"""
def mean_variance_model(
        weights, concentration_c, mode_c):

    concentration = tf.reduce_sum(
        tf.expand_dims(concentration_c, -1) * weights, axis=-2)

    mode = tf.reduce_sum(
        tf.expand_dims(mode_c, -1) * weights, axis=-2)

    # Old nonsensical version
    # scale = 1 / (tf.math.exp(mode) * (concentration + 1))

    scale = (concentration + 1) * mode

    return tfd.InverseGamma(
        concentration=concentration,
        scale=scale)


"""
Choose evenly spaced knots for kernel regression.
"""
def choose_knots(low, high, kernel_regression_degree):
    x_scale_hinges = []
    d = (high - low) / (kernel_regression_degree+1)
    for i in range(kernel_regression_degree):
        x_scale_hinges.append(low + (i+1)*d)
    return x_scale_hinges