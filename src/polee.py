
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.contrib import distributions
from tensorflow.contrib import framework
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed

# models
from polee_transcript_expression import *
from polee_splicing import *
from polee_dropout import *
from polee_transcript_mixture import *
from polee_transcript_vae_mixture import *
from polee_transcript_pca import *
from polee_tsne import *
from polee_linear_regression import *



"""
An improper prior that just returns a log-probability of 0. (I.e. it's not an
actualy distribution)
"""
class ImproperPriorDist(distributions.Distribution):
    def __init__(self, name="ImproperPrior"):
        parameters = locals()

        super(ImproperPriorDist, self).__init__(
            dtype=tf.float32,
            validate_args=False,
            allow_nan_stats=False,
            reparameterization_type=tf.contrib.distributions.FULLY_REPARAMETERIZED,
            parameters=[],
            graph_parents=[])

    def _get_event_shape(self):
        return tf.TensorShape([self._value.get_shape()[-1]])

    def _get_batch_shape(self):
        return self._value.get_shape()[:-1]

    def _log_prob(self, _):
        return tf.zeros([int(self._get_batch_shape()[0])])


class ImproperPrior(ed.RandomVariable, ImproperPriorDist):
    def __init__(self, *args, **kwargs):
        super(ImproperPrior, self).__init__(*args, **kwargs)


"""
A general-purpose hack for incorporating an approximated likelihood function.
This is used particularly for approximated splicing likelihoods.
"""
class ApproximatedLikelihoodDist(distributions.Distribution):
    def __init__(self, dist, x, name="ApproximatedLikelihood"):
        parameters = locals()
        self._dist = dist
        self._x = x

        super(ApproximatedLikelihoodDist, self).__init__(
            dtype=tf.float32,
            validate_args=False,
            allow_nan_stats=False,
            reparameterization_type=tf.contrib.distributions.FULLY_REPARAMETERIZED,
            parameters=[],
            graph_parents=[])

    def _get_event_shape(self):
        return tf.TensorShape([0])

    def _get_batch_shape(self):
        return self._x.get_shape()[:-1]

    def _log_prob(self, _):
        return self._dist._log_prob(self._x)


class ApproximatedLikelihood(ed.RandomVariable, ApproximatedLikelihoodDist):
    def __init__(self, dist, x, *args, **kwargs):
        kwargs["value"] = tf.zeros([x.get_shape()[0], 0])
        super(ApproximatedLikelihood, self).__init__(dist, x, *args, **kwargs)