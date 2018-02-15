
import numpy as np
import tensorflow as tf
from tensorflow.contrib import distributions
from tensorflow.contrib import framework
import edward
from queue import Queue
import sys


class RNASeqApproxLikelihoodDist(distributions.Distribution):
    def __init__(self, x, efflens, la_params, invhsb_params,
                 validate_args=False,
                 allow_nan_stats=False,
                 name="RNASeqApproxLikelihood"):

        with tf.name_scope(name, values=[x]) as ns:
            self.x = tf.identity(x, name="rnaseq/x")
            framework.assert_same_float_dtype([self.x])
        parameters = locals()

        # print(self.x)
        # print(self.x.get_shape())

        # self.x = x
        self.efflens = efflens
        self.la_params = la_params
        self.invhsb_params = invhsb_params

        super(RNASeqApproxLikelihoodDist, self).__init__(
              dtype=self.x.dtype,
              validate_args=validate_args,
              allow_nan_stats=allow_nan_stats,
              reparameterization_type=tf.contrib.distributions.FULLY_REPARAMETERIZED,
              parameters=parameters,
              graph_parents=[self.x,])

    def _get_event_shape(self):
        print("y.get_shape()")
        print(self.x.get_shape())
        return tf.TensorShape([2, self.x.get_shape()[-1] - 1])

    def _get_batch_shape(self):
        print("y.get_shape()")
        print(self.x.get_shape())
        return self.x.get_shape()[:-1]

    def _log_prob(self, _):
        num_samples = int(self.x.get_shape()[0])
        n           = int(self.x.get_shape()[-1])
        num_nodes   = 2*n - 1

        mu    = tf.identity(self.la_params[...,0,:], name="mu")
        sigma = tf.identity(self.la_params[...,1,:], name="sigma")
        alpha = tf.identity(self.la_params[...,2,:], name="alpha")

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
        efflen_ladj = tf.reduce_sum(tf.log(self.efflens), axis=1) - n * tf.log(tf.squeeze(x_scaled_sum))

        # Inverse hierarchical stick breaking transform
        # ---------------------------------------------

        leafindex = self.invhsb_params[0]
        left_child_rightmost_index  = self.invhsb_params[1]
        left_child_leftmost_index   = self.invhsb_params[2]
        right_child_rightmost_index = self.invhsb_params[3]
        right_child_leftmost_index  = self.invhsb_params[4]

        x_permed = tf.gather_nd(x_efflen, leafindex)
        x_permed = tf.to_double(x_permed)

        x_cumsum = tf.cumsum(x_permed, axis=1)
        x_cumsum = tf.concat([tf.zeros([num_samples, 1], tf.float64), x_cumsum], axis=1)

        left_node_values  = tf.log(tf.to_float(tf.gather_nd(x_cumsum, left_child_rightmost_index) -
                                               tf.gather_nd(x_cumsum, left_child_leftmost_index)))
        right_node_values = tf.log(tf.to_float(tf.gather_nd(x_cumsum, right_child_rightmost_index) -
                                               tf.gather_nd(x_cumsum, right_child_leftmost_index)))

        y_logit = tf.identity(left_node_values - right_node_values, name="y_logit")

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

        return lp - efflen_ladj

class RNASeqApproxLikelihood(edward.RandomVariable, RNASeqApproxLikelihoodDist):
    def __init__(self, *args, **kwargs):
        super(RNASeqApproxLikelihood, self).__init__(*args, **kwargs)

