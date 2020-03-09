
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python import framework
import tensorflow_probability as tfp


# Load tensorflow extension for computing stick breaking transformation
polee_src_path = os.path.dirname(os.path.realpath(__file__))
ext_path = os.path.join(polee_src_path, "tensorflow_ext", "hsb_ops.so")
inverse_hsb_op_module = tf.load_op_library(ext_path)

@ops.RegisterGradient("InvHSB")
def _inv_hsb_grad(op, y_grad, ladj_grad):
    left_index  = op.inputs[1]
    right_index = op.inputs[2]
    leaf_index  = op.inputs[3]
    y           = op.outputs[0]
    ladj        = op.outputs[1]

    x_grad = inverse_hsb_op_module.inv_hsb_grad(
        y_grad, ladj_grad, y, ladj, left_index, right_index, leaf_index)

    return [x_grad, None, None, None]


"""
Random expression vector samples in proportion to the approximated likelihood
function.
"""
def rnaseq_approx_likelihood_sampler(
        num_samples, n, efflens, mu, sigma, alpha, left_index, right_index, leaf_index):

    # sampling from likelihood distribution
    z0 = tf.random.normal([num_samples, n-1])

    # sinh-asinh transform
    z = tf.sinh(tf.asinh(z0) + alpha)

    # non-standard-normal transform
    y_logit = mu + sigma * z

    # y_logit = tf.Print(y_logit, [tf.reduce_min(y_logit), tf.reduce_max(y_logit)], "y_logit")

    # hsb transform
    x_efflen = inverse_hsb_op_module.hsb(
        y_logit, left_index, right_index, leaf_index)

    # x_efflen = tf.Print(x_efflen, [tf.reduce_min(x_efflen), tf.reduce_max(x_efflen)], "x_efflen")

    # effective length transform
    x_scaled = x_efflen / efflens
    x = x_scaled / tf.reduce_sum(x_scaled, axis=1, keepdims=True)
    x = tf.clip_by_value(x, 1e-16, 0.99999999e0)
    return x


def rnaseq_approx_likelihood_sampler_from_vars(num_samples, n, vars):
    return rnaseq_approx_likelihood_sampler(
        num_samples, n,
        efflens=vars["efflen"],
        mu=vars["la_mu"],
        sigma=vars["la_sigma"],
        alpha=vars["la_alpha"],
        left_index=vars["left_index"],
        right_index=vars["right_index"],
        leaf_index=vars["leaf_index"])



"""
Approximated RNA-Seq likelihood.
"""
class RNASeqApproxLikelihoodDist(tfp.distributions.Distribution):
    def __init__(self, x, efflens,
                 la_mu,
                 la_sigma,
                 la_alpha,
                 left_index,
                 right_index,
                 leaf_index,
                 name="RNASeqApproxLikelihood"):

        self.x           = x
        self.efflens     = efflens
        self.mu          = la_mu
        self.sigma       = la_sigma
        self.alpha       = la_alpha
        self.left_index  = left_index
        self.right_index = right_index
        self.leaf_index  = leaf_index

        parameters = dict(locals())

        super(RNASeqApproxLikelihoodDist, self).__init__(
              dtype=self.x.dtype,
              reparameterization_type=tfp.distributions.FULLY_REPARAMETERIZED,
              validate_args=False,
              allow_nan_stats=False,
              parameters=parameters,
              name=name)

    def _event_shape(self):
        # return tf.TensorShape([self.x.get_shape()[-1]])
        return tf.TensorShape([])

    def _batch_shape(self):
        return self.x.get_shape()[:-1]

    def _sample_n(self, N, seed=None):
        shape = (N,) + self._batch_shape()
        return tf.zeros(shape)

    # @tf.function
    def _log_prob(self, __ignored__):
        num_samples = int(self.x.get_shape()[-2])
        n           = int(self.x.get_shape()[-1])
        num_nodes   = 2*n - 1

        mu    = self.mu
        sigma = self.sigma
        alpha = self.alpha

        # log absolute determinant of the jacobian
        ladj = 0.0

        x_exp = tf.math.exp(self.x)

        # jacobian for exp transformation
        ladj += tf.reduce_sum(self.x, axis=-1)

        # jacobian for softmax
        ladj -= (n-1) * tf.math.log(tf.reduce_sum(x_exp, axis=-1))

        # compute softmax of x
        x = x_exp / tf.reduce_sum(x_exp, axis=-1, keepdims=True)

        # effective length transform
        # --------------------------

        x_scaled = x * self.efflens
        x_scaled_sum = tf.reduce_sum(x_scaled, axis=-1, keepdims=True)
        x_efflen = x_scaled / x_scaled_sum

        ladj += tf.expand_dims(tf.reduce_sum(tf.math.log(self.efflens), axis=-1), axis=0) - \
                tf.reshape(tf.math.log(x_scaled_sum), [1, num_samples])

        # Inverse hierarchical stick breaking transform
        # ---------------------------------------------

        y_tensors = []
        ptt_ladj_tensors = []
        for x_efflen_batch in tf.unstack(x_efflen):
            y_, ladj_ = inverse_hsb_op_module.inv_hsb(
                    x_efflen_batch,
                    self.left_index, self.right_index, self.leaf_index)

            y_tensors.append(y_)
            ptt_ladj_tensors.append(ladj_)
        y = tf.stack(y_tensors)
        ptt_ladj = tf.stack(ptt_ladj_tensors)
        ladj += tf.reduce_sum(ptt_ladj, axis=-1)

        y_log = tf.math.log(y)
        y_1mlog = tf.math.log1p(-y)

        y_logit = tf.cast(y_log - y_1mlog, tf.float32)

        ladj += tf.reduce_sum(
            tf.cast(-y_log - y_1mlog, tf.float32),
            axis=-1)

        # normal standardization transform
        # --------------------------------

        z_std = tf.divide(tf.subtract(y_logit, mu), sigma)

        ladj += tf.reduce_sum(-tf.math.log(sigma), axis=-1)

        # inverse sinh-asinh transform
        # ----------------------------

        z_asinh = tf.math.asinh(z_std)
        z = tf.sinh(z_asinh - alpha)

        ladj += tf.reduce_sum(
            tf.math.log(tf.math.cosh(alpha - z_asinh)) -
                0.5 * tf.math.log1p(tf.math.square(z_std)),
            axis=-1)

        # standand normal log-probability
        # -------------------------------

        lp = tf.reduce_sum(-np.log(2.0*np.pi) - tf.square(z), axis=-1) / 2.0

        return lp + ladj

def RNASeqApproxLikelihood(*args, **kwargs):
    return RNASeqApproxLikelihoodDist(*args, **kwargs)


def rnaseq_approx_likelihood_log_prob_from_vars(vars, x):
    return tf.reduce_sum(RNASeqApproxLikelihood(
        x=x,
        efflens=vars["efflen"],
        la_mu=vars["la_mu"],
        la_sigma=vars["la_sigma"],
        la_alpha=vars["la_alpha"],
        left_index=vars["left_index"],
        right_index=vars["right_index"],
        leaf_index=vars["leaf_index"]).log_prob())


# class DummyRNASeqLikelihood(tfp.distributions.Distribution):
#     def __init__(self, x, name="DummyRNASeqLikelihood"):
#         self.x = x
#         self.event_shape = tf.TensorShape([2, self.x.get_shape()[-1] - 1])
#         self.batch_shape = self.x.get_shape()[:-1]
#         self.name = "dummy_rnaseq_likelihood"
#         self.dtype = tf.float32
#         self.reparameterization_type=tfp.distributions.FULLY_REPARAMETERIZED

#         # super(RNASeqApproxLikelihoodDist, self).__init__(
#         #       dtype=self.x.dtype,
#         #       validate_args=validate_args,
#         #       allow_nan_stats=allow_nan_stats,
#         #       reparameterization_type=tf.contrib.distributions.FULLY_REPARAMETERIZED,
#         #       parameters=parameters,
#         #       graph_parents=[self.x,])

#     def log_prob(self, __ignored__):
#         return 0.0


def rnaseq_approx_likelihood_from_vars(vars, x):
    return RNASeqApproxLikelihood(
        x=x,
        efflens=vars["efflen"],
        la_mu=vars["la_mu"],
        la_sigma=vars["la_sigma"],
        la_alpha=vars["la_alpha"],
        left_index=vars["left_index"],
        right_index=vars["right_index"],
        leaf_index=vars["leaf_index"])
