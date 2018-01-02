
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


        # self.x = tf.Print(self.x, [tf.reduce_sum(tf.exp(self.x), axis=1)], "X SCALE", summarize=6)

        x = tf.nn.softmax(self.x)

        # effective length transform
        # --------------------------

        x_scaled = x * self.efflens
        x_scaled_sum = tf.reduce_sum(x_scaled, axis=1, keep_dims=True)
        x_efflen = x_scaled / x_scaled_sum

        # efflen_ladj = tf.reduce_sum(tf.log(self.efflens), axis=1) - n * tf.log(tf.squeeze(x_scaled_sum))


        # Inverse hierarchical stick breaking transform
        # ---------------------------------------------

        leafindex = self.invhsb_params[0]
        internal_node_indexes = self.invhsb_params[1]
        internal_node_left_indexes = self.invhsb_params[2]
        leftmost_indexes = self.invhsb_params[3]
        rightmost_indexes = self.invhsb_params[4]

        x_permed = tf.gather_nd(x_efflen, leafindex)
        x_permed = tf.to_double(x_permed)

        x_cumsum = tf.cumsum(x_permed, axis=1)
        x_cumsum = tf.concat([tf.zeros([num_samples, 1], tf.double), x_cumsum], axis=1)

        x_lm = tf.gather_nd(x_cumsum, leftmost_indexes, name="x_lm")
        x_rm = tf.gather_nd(x_cumsum, rightmost_indexes, name="x_rm")

        u = x_rm - x_lm
        u = tf.identity(u, name="u")

        internal_node_values = tf.gather_nd(u, internal_node_indexes)
        left_node_values     = tf.gather_nd(u, internal_node_left_indexes)

        y = tf.divide(left_node_values, internal_node_values, name="y")

        # hsb_ladj = tf.reduce_sum(-tf.log(internal_node_values), axis=1)

        y = tf.clip_by_value(y, 1e-10, 1.0 - 1e-10)


        # logit (inverse logistic) transform
        # ----------------------------------

        y_log = tf.log(y)
        y_om_log = tf.log(1.0 - y)
        y_logit = tf.to_float(y_log - y_om_log)
        # y_logit_ladj = tf.reduce_sum(tf.to_float(-y_log - y_om_log), axis=1)


        # normal standardization transform
        # --------------------------------

        z_std = tf.divide(tf.subtract(y_logit, mu), sigma)
        # z_std_ladj = -tf.reduce_sum(tf.log(sigma), axis=1)

        # inverse sinh-asinh transform
        # ----------------------------

        z_c = tf.subtract(tf.asinh(z_std), alpha)
        z = tf.sinh(z_c)

        z_ladj = tf.reduce_sum(
            tf.subtract(tf.log(tf.cosh(z_c)),
                        tf.multiply(0.5, tf.log1p(tf.square(z_std)))),
            axis=1)

        # standand normal log-probability
        # -------------------------------

        lp = tf.reduce_sum((-np.log(2.0*np.pi) - tf.square(z)) / 2.0, axis=1)

        # TODO: Really there should be jacobian terms, but this seems to only
        # cause problems. I don't know why.
        # return lp + z_ladj + z_std_ladj + y_logit_ladj + \
        #     hsb_ladj + efflen_ladj

        return lp

class RNASeqApproxLikelihood(edward.RandomVariable, RNASeqApproxLikelihoodDist):
    def __init__(self, *args, **kwargs):
        super(RNASeqApproxLikelihood, self).__init__(*args, **kwargs)

