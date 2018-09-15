
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.contrib import distributions
from tensorflow.contrib import framework
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed


# Load tensorflow extension for computing stick breaking transformation
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


"""
Random expression vector samples in proportion to the approximated likelihood
function.
"""
def rnaseq_approx_likelihood_sampler(efflens, la_params, hsb_params):
    mu    = tf.identity(la_params[...,0,:], name="mu")
    sigma = tf.identity(la_params[...,1,:], name="sigma")
    alpha = tf.identity(la_params[...,2,:], name="alpha")

    left_index  = hsb_params[0]
    right_index = hsb_params[1]
    leaf_index  = hsb_params[2]

    # sampling from likelihood distribution
    z0 = ed.models.Normal(loc=tf.zeros(mu.get_shape()), scale=[1.0])

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
    x = tf.clip_by_value(x, 1e-15, 0.9999999e0)
    return x


"""
Approximated RNA-Seq likelihood.
"""
class RNASeqApproxLikelihoodDist(distributions.Distribution):
    def __init__(self, x, efflens,
                 la_mu,
                 la_sigma,
                 la_alpha,
                 left_index,
                 right_index,
                 leaf_index,
                 informative_prior=False,
                 validate_args=False,
                 allow_nan_stats=False,
                 name="RNASeqApproxLikelihood"):

        with tf.name_scope(name, values=[x]) as ns:
            self.x = tf.identity(x, name="rnaseq/x")
            framework.assert_same_float_dtype([self.x])
        parameters = locals()

        self.efflens = efflens

        self.mu    = tf.identity(la_mu,    name="likapr_mu")
        self.sigma = tf.identity(la_sigma, name="likapr_sigma")
        self.alpha = tf.identity(la_alpha, name="likapr_alpha")

        self.left_index  = left_index
        self.right_index = right_index
        self.leaf_index  = leaf_index

        self.informative_prior = informative_prior

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

        # x= tf.Print(x, [1e6 * x[1 - 1, 199451 - 1]], "x: ")

        # TODO: This is probably not a good idea. It tends to make solutions
        # pretty unstable (introduces a lot of locat maxima that are hard to
        # escape, I suspect)
        # optional LogNormal regularization essentially encodes the assumption
        # that most transcripts are not expressed.
        # these prior values are just sort of rules of thumb that seem to do ok
        if self.informative_prior:
            prior_mean = np.log(1e-5 * 1/n)
            prior_var = (np.log(1/n) - prior_mean)
            x_prior_lp = \
                -tf.reduce_sum(tf.log(x), axis=1) + \
                0.5 * (-np.log(2.0*np.pi*prior_var) - \
                tf.reduce_sum(tf.square(tf.log(x) - prior_mean) / prior_var, axis=1))
        else:
            x_prior_lp = 0.0

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

        return lp + x_prior_lp

def RNASeqApproxLikelihood(*args, **kwargs):
    return RNASeqApproxLikelihoodDist(*args, **kwargs)


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
