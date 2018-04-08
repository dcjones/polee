
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.framework import ops
from tensorflow.contrib import distributions
from tensorflow.contrib import framework
import edward
from queue import Queue
import sys

polee_src_path = os.path.dirname(os.path.realpath(__file__))
ext_path = os.path.join(polee_src_path, "tensorflow_ext", "hsb_ops.so")
inverse_hsb_op_module = tf.load_op_library(ext_path)

@ops.RegisterGradient("InvHSB")
def _inv_hsb_grad(op, grad):
    left_index  = op.inputs[1]
    right_index = op.inputs[2]
    leaf_index  = op.inputs[3]
    y_logit     = op.outputs[0]

    x_grad = inverse_hsb_op_module.inv_hsb_grad(grad, y_logit, left_index,
                                                 right_index, leaf_index)
    return [x_grad, None, None, None]


def rnaseq_approx_likelihood_sampler(efflens, la_params, hsb_params):
    mu    = tf.identity(la_params[...,0,:], name="mu")
    sigma = tf.identity(la_params[...,1,:], name="sigma")
    alpha = tf.identity(la_params[...,2,:], name="alpha")

    left_index  = hsb_params[0]
    right_index = hsb_params[1]
    leaf_index  = hsb_params[2]

    # sampling from likelihood distribution
    z0 = edward.models.Normal(loc=tf.zeros(mu.get_shape()), scale=[1.0])

    # sinh-asinh transform
    z = tf.sinh(tf.asinh(z0) + alpha)

    # non-standard-normal transform
    y_logit = mu + sigma * z

    y_logit = tf.Print(y_logit, [tf.reduce_min(y_logit), tf.reduce_max(y_logit)], "y_logit")

    # hsb transform
    x_efflen = inverse_hsb_op_module.hsb(
        y_logit, left_index, right_index, leaf_index)

    x_efflen = tf.Print(x_efflen, [tf.reduce_min(x_efflen), tf.reduce_max(x_efflen)], "x_efflen")

    # effective length transform
    x_scaled = x_efflen / efflens
    x = x_scaled / tf.reduce_sum(x_scaled, axis=1, keepdims=True)
    x = tf.clip_by_value(x, 1.1920929e-7, 0.9999999e0)
    return x


class RNASeqApproxLikelihoodDist(distributions.Distribution):
    def __init__(self, x, efflens, la_params, invhsb_params,
                 validate_args=False,
                 allow_nan_stats=False,
                 name="RNASeqApproxLikelihood"):

        with tf.name_scope(name, values=[x]) as ns:
            self.x = tf.identity(x, name="rnaseq/x")
            framework.assert_same_float_dtype([self.x])
        parameters = locals()

        self.efflens = efflens

        self.mu    = tf.identity(la_params[...,0,:], name="mu")
        self.sigma = tf.identity(la_params[...,1,:], name="sigma")
        self.alpha = tf.identity(la_params[...,2,:], name="alpha")

        self.left_index  = invhsb_params[0]
        self.right_index = invhsb_params[1]
        self.leaf_index  = invhsb_params[2]


        super(RNASeqApproxLikelihoodDist, self).__init__(
              dtype=self.x.dtype,
              validate_args=validate_args,
              allow_nan_stats=allow_nan_stats,
              reparameterization_type=tf.contrib.distributions.FULLY_REPARAMETERIZED,
              parameters=parameters,
              graph_parents=[self.x,])

    def _get_event_shape(self):
        return tf.TensorShape([2, self.x.get_shape()[-1] - 1])

    def _get_batch_shape(self):
        return self.x.get_shape()[:-1]

    def _log_prob(self, _):
        num_samples = int(self.x.get_shape()[0])
        n           = int(self.x.get_shape()[-1])
        num_nodes   = 2*n - 1

        mu    = self.mu
        sigma = self.sigma
        alpha = self.alpha

        # self.x = tf.Print(self.x, [tf.reduce_sum(tf.exp(self.x), axis=1)], "x scale")
        x = tf.nn.softmax(self.x)

        # effective length transform
        # --------------------------

        efflens = self.efflens
        x_scaled = x * efflens
        x_scaled_sum = tf.reduce_sum(x_scaled, axis=1, keepdims=True)
        x_efflen = x_scaled / x_scaled_sum

        # Inverse hierarchical stick breaking transform
        # ---------------------------------------------

        y_logit = inverse_hsb_op_module.inv_hsb(
            x_efflen, self.left_index, self.right_index, self.leaf_index)

        # normal standardization transform
        # --------------------------------

        z_std = tf.divide(tf.subtract(y_logit, mu), sigma)

        # inverse sinh-asinh transform
        # ----------------------------

        z_c = tf.subtract(tf.asinh(z_std), alpha)
        z = tf.sinh(z_c)

        # standand normal log-probability
        # -------------------------------

        lp = (-np.log(2.0*np.pi) -  tf.reduce_sum(tf.square(z), axis=1)) / 2.0

        return lp

class RNASeqApproxLikelihood(edward.RandomVariable, RNASeqApproxLikelihoodDist):
    def __init__(self, *args, **kwargs):
        super(RNASeqApproxLikelihood, self).__init__(*args, **kwargs)


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


class ImproperPrior(edward.RandomVariable, ImproperPriorDist):
    def __init__(self, *args, **kwargs):
        super(ImproperPrior, self).__init__(*args, **kwargs)


# A more general way (hack) of dealing with approximated likelihood functions.
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


class ApproximatedLikelihood(edward.RandomVariable, ApproximatedLikelihoodDist):
    def __init__(self, dist, x, *args, **kwargs):
        kwargs["value"] = tf.zeros([x.get_shape()[0], 0])
        super(ApproximatedLikelihood, self).__init__(dist, x, *args, **kwargs)