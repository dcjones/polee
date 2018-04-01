
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.contrib import distributions
from tensorflow.contrib import framework
import edward
from queue import Queue
import sys

# TODO: compute this path
inverse_hsb_op_module = tf.load_op_library("/home/dcjones/prj/extruder/src/tensorflow_ext/hsb_ops.so")

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

    # hsb transform
    x_efflen = inverse_hsb_op_module.hsb(
        y_logit, left_index, right_index, leaf_index)

    # effective length transform
    x_scaled = x_efflen / efflens
    x = x_scaled / tf.reduce_sum(x_scaled, axis=1, keepdims=True)
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

        # Approximated likelihood assumes a uniform prior over x * efflens. We
        # want instead a uniform prior over x (i.e. a function proportional to
        # the likelihood). To get that, we correct the approximate likelihood
        # using the log absolute determinant of the jacobian for the effective
        # length transformation.
        # efflen_ladj = tf.reduce_sum(tf.log(self.efflens), axis=1) - n * tf.log(tf.squeeze(x_scaled_sum))

        # Inverse hierarchical stick breaking transform
        # ---------------------------------------------

        # leafindex = self.invhsb_params[0]
        # left_child_rightmost_index  = self.invhsb_params[1]
        # left_child_leftmost_index   = self.invhsb_params[2]
        # right_child_rightmost_index = self.invhsb_params[3]
        # right_child_leftmost_index  = self.invhsb_params[4]

        # x_permed = tf.gather_nd(x_efflen, leafindex)
        # x_permed = tf.to_double(x_permed)

        # x_cumsum = tf.cumsum(x_permed, axis=1)
        # x_cumsum = tf.concat([tf.zeros([num_samples, 1], tf.float64), x_cumsum], axis=1)

        # left_node_values  = tf.log(tf.to_float(tf.gather_nd(x_cumsum, left_child_rightmost_index) -
        #                                        tf.gather_nd(x_cumsum, left_child_leftmost_index)))
        # right_node_values = tf.log(tf.to_float(tf.gather_nd(x_cumsum, right_child_rightmost_index) -
        #                                        tf.gather_nd(x_cumsum, right_child_leftmost_index)))

        # y_logit = tf.identity(left_node_values - right_node_values, name="y_logit")

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


# Values for gate_gradients.
GATE_NONE = 0
GATE_OP = 1
GATE_GRAPH = 2

# Taken from: https://github.com/blei-lab/edward/issues/708
class ClippedAdamOptimizer(tf.train.AdamOptimizer):
    """
    Clipped version adam optimizer, where its gradient is clipped by value
    so that it cannot be too large.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-08, use_locking=False,
                 clip_func=lambda x: tf.clip_by_value(x, 1.0e-4, 1.0e4),
                 name='Adam'):
        super(ClippedAdamOptimizer, self).__init__(
            learning_rate, beta1, beta2, epsilon, use_locking, name)
        self._clip_func = clip_func

    def compute_gradients(self, loss, var_list=None, gate_gradients=GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False, grad_loss=None):
        print("ClippedAdamOptimizer compute_gradients")
        sys.exit()
        grad_and_vars = super(ClippedAdamOptimizer, self).compute_gradients(
            loss, var_list, gate_gradients, aggregation_method,
            colocate_gradients_with_ops, grad_loss)

        for g in grad_and_vars.keys():
            print(g)
            v = grad_and_vars[v]
            grad_and_vars[g] = tf.Print(v, [tf.reduce_min(v), tf.reduce_max(v)], "grad extrema")

        # clip func
        if self._clip_func is None:
            return grad_and_vars
        return [(self._clip_func(g, v) if g is not None else (g, v)
                for g, v in grad_and_vars)]


class ClippedKumaraswamy(edward.models.Kumaraswamy):
    def __init__(self, alpha, beta, name="ClippedKumaraswamy"):
        super(ClippedKumaraswamy, self).__init__(alpha, beta, name=name)
        # self._value = tf.clip_by_value(self._value, 1e-7, 1 - 1e-7)
        self._value = tf.clip_by_value(self._value, 1e-2, 1 - 1e-2)


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
