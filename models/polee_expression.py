
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from polee_approx_likelihood import *
from polee import *

"""
Basic shrinkage model for making unsupervised transcript expression estimates.

This is similar to the regression model, but there is no regression coefficients
and we just model expression directly using a horseshoe prior.
"""
class RNASeqExpression:
    def __init__(self, x_init, likelihood_model, surrogate_likelihood_model,
            x_bias_loc0, x_bias_scale0, sample_scales, use_point_estimates):

        self.num_samples= int(x_init.shape[0])
        self.num_features = int(x_init.shape[1])

        self.likelihood_model = likelihood_model
        self.surrogate_likelihood_model = surrogate_likelihood_model
        self.x_bias_loc0 = x_bias_loc0
        self.x_bias_scale0 = x_bias_scale0
        self.sample_scales = sample_scales
        self.use_point_estimates = use_point_estimates

        self.qx_global_scale_variance_loc_var = tf.Variable(0.0)
        self.qx_global_scale_variance_softplus_scale_var = tf.Variable(-1.0)

        self.qx_global_scale_noncentered_loc_var = tf.Variable(0.0)
        self.qx_global_scale_noncentered_softplus_scale_var = tf.Variable(-1.0)

        self.qx_local1_scale_variance_loc_var = tf.Variable(
            tf.fill([self.num_samples, self.num_features], 0.0))
        self.qx_local1_scale_variance_softplus_scale_var = tf.Variable(
            tf.fill([self.num_samples, self.num_features], -1.0))

        self.qx_local1_scale_noncentered_loc_var = tf.Variable(
            tf.fill([self.num_samples, self.num_features], 0.0))
        self.qx_local1_scale_noncentered_softplus_scale_var = tf.Variable(
            tf.fill([self.num_samples, self.num_features], -1.0))

        self.x_bias_init = tf.reduce_mean(x_init, axis=0)

        self.qx_bias_loc_var = tf.Variable(self.x_bias_init)
        self.qx_bias_softplus_scale_var = tf.Variable(
            tf.fill([self.num_features], -1.0))

        self.qx_loc_var = tf.Variable(
            x_init,
            trainable=not use_point_estimates)

        self.qx_softplus_scale_var = tf.Variable(
            tf.fill([self.num_samples, self.num_features], -1.0))


    def model_fn(self, likelihood_model, sample_scales):
        x_global_scale_variance = yield JDCRoot(Independent(tfd.InverseGamma(
            concentration=0.5, scale=0.5)))
        x_global_scale_noncentered = yield JDCRoot(Independent(tfd.HalfNormal(
            scale=1.0)))
        x_global_scale = x_global_scale_noncentered * tf.sqrt(x_global_scale_variance)

        x_local1_scale_variance = yield JDCRoot(Independent(tfd.InverseGamma(
            concentration=tf.fill([self.num_samples, self.num_features], 0.5),
            scale=tf.fill([self.num_samples, self.num_features], 0.5))))
        x_local1_scale_noncentered = yield JDCRoot(Independent(tfd.HalfNormal(
            scale=tf.ones([self.num_samples, self.num_features]))))
        x_local1_scale = x_local1_scale_noncentered * tf.sqrt(x_local1_scale_variance)

        x_bias = yield JDCRoot(Independent(tfd.Normal(
            loc=tf.fill([self.num_features], np.float32(self.x_bias_loc0)),
            scale=np.float32(self.x_bias_scale0))))

        x = yield Independent(tfd.Normal(
            loc=x_bias - sample_scales,
            scale=x_local1_scale * x_global_scale))

        yield from likelihood_model(x)


    def variational_model_fn(self, surrogate_likelihood_model):
        qx_global_scale_variance = yield JDCRoot(Independent(SoftplusNormal(
            loc=self.qx_global_scale_variance_loc_var,
            scale=tf.nn.softplus(self.qx_global_scale_variance_softplus_scale_var))))

        qx_global_scale_noncentered = yield JDCRoot(Independent(SoftplusNormal(
            loc=self.qx_global_scale_noncentered_loc_var,
            scale=tf.nn.softplus(self.qx_global_scale_noncentered_softplus_scale_var))))

        qx_local1_scale_variance = yield JDCRoot(Independent(SoftplusNormal(
            loc=self.qx_local1_scale_variance_loc_var,
            scale=tf.nn.softplus(self.qx_local1_scale_variance_softplus_scale_var))))

        qx_local1_scale_noncentered = yield JDCRoot(Independent(SoftplusNormal(
            loc=self.qx_local1_scale_noncentered_loc_var,
            scale=tf.nn.softplus(self.qx_local1_scale_noncentered_softplus_scale_var))))

        qx_bias = yield JDCRoot(Independent(tfd.Normal(
            loc=self.qx_bias_loc_var,
            scale=tf.nn.softplus(self.qx_bias_softplus_scale_var))))

        if self.use_point_estimates:
            qx = yield JDCRoot(Independent(tfd.Deterministic(
                loc=self.qx_loc_var)))
        else:
            qx = yield JDCRoot(Independent(tfd.Normal(
                loc=self.qx_loc_var,
                scale=tf.nn.softplus(self.qx_softplus_scale_var))))

        yield from surrogate_likelihood_model(qx)


    def fit(self, niter):
        model = tfd.JointDistributionCoroutine(
            lambda: self.model_fn(
                self.likelihood_model,
                self.sample_scales))

        variational_model = tfd.JointDistributionCoroutine(
            lambda: self.variational_model_fn(
                self.surrogate_likelihood_model))

        step_num = tf.Variable(1, trainable=False)

        # @tf.function
        def trace_fn(loss, grad, vars):
            def doprint():
                tf.print("[", step_num, "/", niter, "]  loss: ", loss, sep='')
                return tf.constant(0)

            def dontprint():
                return tf.constant(0)

            tf.cond(
                tf.math.mod(step_num, 200) == 0,
                doprint,
                dontprint)

            step_num.assign(step_num + 1)
            return loss

        trace = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn=lambda *args: model.log_prob(args),
            surrogate_posterior=variational_model,
            optimizer=tf.optimizers.Adam(learning_rate=2e-3),
            sample_size=1,
            num_steps=niter,
            trace_fn=trace_fn)

        return self.qx_loc_var.numpy(), tf.nn.softplus(self.qx_softplus_scale_var).numpy()


class RNASeqTranscriptExpression(RNASeqExpression):
    def __init__(self, vars, x_init, sample_scales, use_point_estimates):
        num_samples = x_init.shape[0]
        num_features = x_init.shape[1]

        x_init_mean = np.mean(x_init, axis=0)
        x_bias_loc0 = np.log(1/num_features)
        x_bias_scale0 = 12.0

        def likelihood_model(x):
            if not use_point_estimates:
                likelihood = yield tfd.Independent(
                    rnaseq_approx_likelihood_from_vars(vars, x))

        def surrogate_likelihood_model(qx):
            if not use_point_estimates:
                qrnaseq_reads = yield JDCRoot(Independent(
                    tfd.Deterministic(tf.zeros([num_samples, 0]))))

        super(RNASeqTranscriptExpression, self).__init__(
            x_init, likelihood_model, surrogate_likelihood_model,
            x_bias_loc0, x_bias_scale0, sample_scales, use_point_estimates)

