
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import sys
from tensorflow_probability import edward2 as ed
from polee import *
from polee_approx_likelihood import *
from polee_training import *
import polee_regression


class RNASeqPCA(polee_regression.RNASeqLinearRegression):
    def __init__(
            self, vars, x_init,
            sample_scales, use_point_estimates,
            latent_dimensionality=2,
            kernel_regression_degree=15, kernel_regression_bandwidth=1.0,
            niter=6000):

        self.latent_dimensionality = latent_dimensionality
        num_samples = x_init.shape[0]
        num_features = x_init.shape[1]

        x_init_mean = np.mean(x_init, axis=0)
        x_scale_hinges = tf.constant(
            choose_knots(np.min(x_init_mean), np.max(x_init_mean), kernel_regression_degree),
            dtype=tf.float32)

        x_bias_mu0 = np.log(1/num_features)
        x_bias_sigma0 = 12.0

        self.qz_loc_var = tf.Variable(tf.zeros([num_samples, latent_dimensionality]))
        self.qz_softplus_scale_var = tf.Variable(tf.fill([num_samples, latent_dimensionality], -1.0))

        def likelihood_model(x):
            if not use_point_estimates:
                likelihood = yield tfd.Independent(rnaseq_approx_likelihood_from_vars(vars, x))

        def surrogate_likelihood_model(qx):
            if not use_point_estimates:
                qrnaseq_reads = yield JDCRoot(Independent(tfd.Deterministic(tf.zeros([num_samples]))))

        super(RNASeqPCA, self).__init__(
            self.qz_loc_var, x_init, likelihood_model, surrogate_likelihood_model,
            x_bias_mu0, x_bias_sigma0, x_scale_hinges, sample_scales,
            use_point_estimates,
            kernel_regression_degree, kernel_regression_bandwidth)

        self.qw_loc_var.assign(0.01 * tf.random.normal([self.num_factors, self.num_features]))

    def latent_space_model_fn(self):
        z = yield JDCRoot(Independent(tfd.Normal(
            loc=tf.zeros([self.num_samples, self.latent_dimensionality]),
            scale=1.0)))
        return z

    def surrogate_latent_space_model_fn(self, qz_loc_var, qz_softplus_scale_var):
        qz = yield JDCRoot(Independent(tfd.Deterministic(loc=qz_loc_var)))

    def fit(self, niter):
        model = tfd.JointDistributionCoroutine(
            lambda: self.model_fn(
                self.latent_space_model_fn,
                self.likelihood_model,
                self.sample_scales))

        variational_model = tfd.JointDistributionCoroutine(
            lambda: self.variational_model_fn(
                lambda: self.surrogate_latent_space_model_fn(
                    self.qz_loc_var,
                    self.qz_softplus_scale_var),
                self.surrogate_likelihood_model))

        step_num = tf.Variable(1, trainable=False)

        @tf.function
        def trace_fn(loss, grad, vars):
            if tf.math.mod(step_num, 200) == 0:
                tf.print("[", step_num, "/", niter, "]  loss: ", loss, sep='')
            step_num.assign(step_num + 1)
            return loss

        trace = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn=lambda *args: model.log_prob(args),
            surrogate_posterior=variational_model,
            optimizer=tf.optimizers.Adam(learning_rate=1e-3),
            sample_size=1,
            num_steps=niter,
            trace_fn=trace_fn)

        return (self.qz_loc_var.numpy(), self.qw_loc_var.numpy())

    """
    Take a new set of samples and apply the fit model, optimizing the latent
    positions of the new samples.
    """
    def generalize(sels):
        pass