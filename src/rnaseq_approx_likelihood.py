
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


        # self.x = tf.Print(self.x, [tf.reduce_sum(tf.exp(self.x), axis=1)], "X scale", summarize=6)
        # self.x = tf.Print(self.x, [tf.reduce_min(self.x), tf.reduce_max(self.x)], "X span")

        x = tf.nn.softmax(self.x)

        # effective length transform
        # --------------------------

        efflens = self.efflens
        # efflens = tf.clip_by_value(efflens, 200.0, 1e9)
        # efflens = tf.Print(efflens, [tf.reduce_min(efflens), tf.reduce_max(efflens)], "efflens scale")
        x_scaled = x * efflens
        # x_scaled = x * self.efflens
        # x_scaled = tf.Print(x_scaled, [tf.reduce_min(x_scaled), tf.reduce_max(x_scaled)], "X_SCALED SCALE")
        x_scaled_sum = tf.reduce_sum(x_scaled, axis=1, keep_dims=True)
        x_efflen = x_scaled / x_scaled_sum

        # x_scaled = tf.to_double(x * self.efflens)
        # x_scaled_sum = tf.reduce_sum(x_scaled, axis=1, keep_dims=True)
        # x_efflen = tf.to_float(x_scaled / x_scaled_sum)

        # x_efflen = x

        # I think it's actually the negation of this
        # efflen_ladj = tf.reduce_sum(tf.log(self.efflens), axis=1) - n * tf.log(tf.squeeze(x_scaled_sum))


        # Inverse hierarchical stick breaking transform
        # ---------------------------------------------

        leafindex = self.invhsb_params[0]
        internal_node_left_indexes = self.invhsb_params[1]
        internal_node_right_indexes = self.invhsb_params[2]
        leftmost_indexes = self.invhsb_params[3]
        rightmost_indexes = self.invhsb_params[4]

        x_permed = tf.gather_nd(x_efflen, leafindex)

        # 31-bit fixed point
        # This version convers numbers to 31bit fixed point to do cumsum and
        # ensuring calculations

            # x = tf.Print(x, [tf.reduce_min(x), tf.reduce_max(x)], "X SPAN")

            # smallest number that can be represented in 31-bit fixed point
            # fixed32_eps = 4.656613e-10

            # x_permed = tf.clip_by_value(x_permed, fixed32_eps, 1.0)

            # x_fixed = tf.to_int32(tf.round(x_permed * 2**31)) - 1 # to fixed

            # x_fixed = tf.Print(x_fixed, [tf.reduce_min(x_fixed), tf.reduce_max(x_fixed)], "X FIXED SPAN")

            # x_cumsum = tf.cumsum(x_fixed, axis=1)
            # x_cumsum = tf.concat([tf.zeros([num_samples, 1], tf.int32), x_cumsum], axis=1)

            # x_cumsum = tf.Print(x_cumsum, [tf.reduce_min(x_cumsum), tf.reduce_max(x_cumsum)], "X CUMSUM SPAN")

            # x_lm = tf.gather_nd(x_cumsum, leftmost_indexes, name="x_lm")
            # x_rm = tf.gather_nd(x_cumsum, rightmost_indexes, name="x_rm")
            # u_fixed = x_rm - x_lm
            # u = tf.to_float(u_fixed + 1) / 2**31
            # u_log = tf.log(u)

            # u_log = tf.Print(u_log, [tf.reduce_min(u_log), tf.reduce_max(u_log)], "U LOG SPAN")

            # internal_node_values  = tf.gather_nd(u_log, internal_node_indexes)
            # left_node_values      = tf.gather_nd(u_log, internal_node_left_indexes)
            # right_node_values     = tf.gather_nd(u_log, internal_node_right_indexes)

            # y_logit = tf.divide(left_node_values - internal_node_values,
            #                     right_node_values - internal_node_values )

            # tmp = left_node_values - internal_node_values
            # y_logit = tf.Print(y_logit, [tf.reduce_min(tmp), tf.reduce_max(tmp)], "TMP SPAN")

        # double precision
        # This version is probably simpler and safer, but uses al ot of memory
        # and time by doing everything with float64

        x_permed = tf.to_double(x_permed)

        x_cumsum = tf.cumsum(x_permed, axis=1)
        x_cumsum = tf.concat([tf.zeros([num_samples, 1], tf.float64), x_cumsum], axis=1)

        x_lm = tf.gather_nd(x_cumsum, leftmost_indexes, name="x_lm")
        x_rm = tf.gather_nd(x_cumsum, rightmost_indexes, name="x_rm")

        u = tf.identity(x_rm - x_lm, name="u")
        u = tf.to_float(u)
        u_log = tf.log(u, name="u_log")

        left_node_values  = tf.gather_nd(u_log, internal_node_left_indexes)
        right_node_values = tf.gather_nd(u_log, internal_node_right_indexes)

        y_logit = tf.identity(left_node_values - right_node_values, name="y_logit")
        # y_logit = tf.Print(y_logit, [tf.reduce_min(y_logit), tf.reduce_max(y_logit)], "y_logit span")

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

        lp = (-np.log(2.0*np.pi) -  tf.reduce_sum(tf.square(z), axis=1)) / 2.0

        # TODO: Really there should be jacobian terms, but this seems to only
        # cause problems. I don't know why.
        # return lp + z_ladj + z_std_ladj + y_logit_ladj + \
        #     hsb_ladj + efflen_ladj

        return lp

class RNASeqApproxLikelihood(edward.RandomVariable, RNASeqApproxLikelihoodDist):
    def __init__(self, *args, **kwargs):
        super(RNASeqApproxLikelihood, self).__init__(*args, **kwargs)

