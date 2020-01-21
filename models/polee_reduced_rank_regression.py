
"""
Linear regression with a regression coefficients factored into a low
dimensional space. kind of like doing a regression then PCA on the coefficients,
but in one model.
"""


import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import sys
from polee_approx_likelihood import *
from polee import *


class RNASeqReducedRankRegression:
    def __init__(
        self, k, vars, x_init, F_arr, confounders, sample_scales, use_point_estimates,
        kernel_regression_degree=15, kernel_regression_bandwidth=1.0):

        num_factors = F_arr.shape[1]
        if confounders is not None:
            num_factors += confounders.shape[1]
            self.num_confounders = int(confounders.shape[1])
        else:
            self.num_confounders = 0

        self.num_samples = len(sample_scales)
        self.num_training_samples = int(F_arr.shape[0])
        self.num_testing_samples = self.num_samples - self.num_training_samples
        self.confounders = confounders
        # TODO: include confounders in the model
        self.num_factors = num_factors
        self.k = k # latent dimensionality
        self.F = tf.constant(F_arr, dtype=tf.float32)
        self.vars = vars
        self.use_point_estimates = use_point_estimates
        self.sample_scales = sample_scales

        self.num_features = x_init.shape[1]
        self.x_bias_loc0 = np.log(1/self.num_features)
        self.x_bias_scale0 = 12.0

        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.layers.Dense(20, activation=tf.nn.leaky_relu))
        self.decoder.add(tf.keras.layers.Dense(20, activation=tf.nn.leaky_relu))
        self.decoder.add(tf.keras.layers.Dense(20, activation=tf.nn.leaky_relu))
        self.decoder.add(tf.keras.layers.Dense(self.num_features, activation=tf.identity))

        x_init_mean = np.mean(x_init, axis=0)
        self.x_scale_hinges = tf.constant(
            choose_knots(np.min(x_init_mean), np.max(x_init_mean), kernel_regression_degree),
            dtype=tf.float32)
        self.kernel_regression_bandwidth = kernel_regression_bandwidth
        self.kernel_regression_degree = kernel_regression_degree

        # variational model parameters
        self.qw_loc_var = tf.Variable(
            tf.random.normal([self.num_factors, self.k]))
        self.qw_softplus_scale_var = tf.Variable(
            tf.fill([self.num_factors, self.k], -1.0))

        self.qz_scale_loc_var = tf.Variable(
            tf.fill([self.num_factors, self.k], 0.0))
        self.qz_scale_softplus_scale_var = tf.Variable(
            tf.fill([self.num_factors, self.k], -1.0))

        self.qF_logits_var = tf.Variable(tf.zeros([
            self.num_testing_samples, self.num_factors - self.num_confounders]))

        self.init_temp = 1.0
        self.qF_temperature_var = tf.Variable(self.init_temp, trainable=False)

        F_test =tf.zeros([self.num_testing_samples, self.num_factors - self.num_confounders], tf.float32)
        F_full = tf.concat([self.F, F_test], axis=-2)
        self.qz_loc_var = tf.Variable(
            tf.matmul(F_full, self.qw_loc_var))
            # tf.random.normal([self.num_samples, self.k]))
        self.qz_softplus_scale_var = tf.Variable(
            tf.fill([self.num_samples, self.k], -1.0))

        self.qx_bias_loc_var = tf.Variable(
            tf.reduce_mean(x_init, axis=0))
        self.qx_bias_softplus_scale_var = tf.Variable(
            tf.fill([self.num_features], -1.0))

        self.qx_scale_concentration_c_loc_var = tf.Variable(
            tf.zeros([self.kernel_regression_degree]))

        self.qx_scale_mode_c_loc_var = tf.Variable(
            tf.zeros([self.kernel_regression_degree]))

        self.qx_scale_loc_var = tf.Variable(
            tf.fill([self.num_features], 0.0))
        self.qx_scale_softplus_scale_var = tf.Variable(
            tf.fill([self.num_features], -1.0))

        self.qx_loc_var = tf.Variable(
            x_init,
            trainable=not use_point_estimates)
        self.qx_softplus_scale_var = tf.Variable(
            tf.fill([self.num_samples, self.num_features], 0.0))

    def model_fn(self):
        # regression in latent space
        w = yield JDCRoot(Independent(tfd.Normal(
            loc=tf.zeros([self.num_factors, self.k]),
            scale=tf.fill([self.num_factors, self.k], 10.0))))

        z_scale = yield JDCRoot(Independent(tfd.HalfCauchy(
            loc=tf.zeros([self.num_factors, self.k]),
            scale=1.0)))

        F_test = yield JDCRoot(Independent(tfd.OneHotCategorical(
            logits=tf.zeros([
                self.num_testing_samples, self.num_factors - self.num_confounders]))))

        F_full = tf.concat([tf.expand_dims(self.F, 0), F_test], axis=-2)

        z = yield Independent(tfd.Normal(
            loc=tf.matmul(F_full, w),
            scale=tf.matmul(F_full, z_scale)))

        x_bias = yield JDCRoot(Independent(tfd.Normal(
            loc=tf.fill([self.num_features], np.float32(self.x_bias_loc0)),
            scale=np.float32(self.x_bias_scale0))))

        # decoded log-expression space
        x_loc = x_bias + self.decoder(z) - self.sample_scales

        x_scale_concentration_c = yield JDCRoot(Independent(tfd.HalfCauchy(
            loc=tf.zeros([self.kernel_regression_degree]), scale=1.0)))

        x_scale_mode_c = yield JDCRoot(Independent(tfd.HalfCauchy(
            loc=tf.zeros([self.kernel_regression_degree]), scale=1.0)))

        weights = kernel_regression_weights(
            self.kernel_regression_bandwidth, x_bias, self.x_scale_hinges)

        x_scale = yield Independent(mean_variance_model(
            weights, x_scale_concentration_c, x_scale_mode_c))

        # log expression distribution
        x = yield Independent(tfd.StudentT(
            df=1.0,
            loc=x_loc,
            scale=x_scale))

        if not self.use_point_estimates:
            rnaseq_reads = yield tfd.Independent(rnaseq_approx_likelihood_from_vars(self.vars, x))


    def variational_model_fn(self):
        qw = yield JDCRoot(Independent(tfd.Normal(
            loc=self.qw_loc_var,
            scale=tf.nn.softplus(self.qw_softplus_scale_var))))

        qz_scale = yield JDCRoot(Independent(SoftplusNormal(
            loc=self.qz_scale_loc_var,
            scale=tf.nn.softplus(self.qz_scale_softplus_scale_var))))

        qF_test = yield JDCRoot(Independent(tfd.RelaxedOneHotCategorical(
            temperature=self.qF_temperature_var,
            logits=self.qF_logits_var)))

        # qz = yield JDCRoot(Independent(tfd.Normal(
        #     loc=qz_loc_var,
        #     scale=tf.nn.softplus(qz_softplus_scale_var))))
        qz = yield JDCRoot(Independent(tfd.Deterministic(
            loc=self.qz_loc_var)))

        qx_bias = yield JDCRoot(Independent(tfd.Normal(
            loc=self.qx_bias_loc_var,
            scale=tf.nn.softplus(self.qx_bias_softplus_scale_var))))

        qx_scale_concentration_c = yield JDCRoot(Independent(tfd.Deterministic(
            loc=tf.nn.softplus(self.qx_scale_concentration_c_loc_var))))

        qx_scale_mode_c = yield JDCRoot(Independent(tfd.Deterministic(
            loc=tf.nn.softplus(self.qx_scale_mode_c_loc_var))))

        qx_scale = yield JDCRoot(Independent(SoftplusNormal(
            loc=self.qx_scale_loc_var,
            scale=tf.nn.softplus(self.qx_scale_softplus_scale_var))))

        if self.use_point_estimates:
            qx = yield JDCRoot(Independent(tfd.Deterministic(
                loc=self.qx_loc_var)))
        else:
            qx = yield JDCRoot(Independent(tfd.Normal(
                loc=self.qx_loc_var,
                scale=tf.nn.softplus(self.qx_softplus_scale_var))))
            qrnaseq_reads = yield JDCRoot(Independent(
                tfd.Deterministic(tf.zeros([self.num_samples]))))


    def fit(self, niter):

        model = tfd.JointDistributionCoroutine(
            lambda: self.model_fn())

        variational_model = tfd.JointDistributionCoroutine(
            lambda: self.variational_model_fn())

        step_num = tf.Variable(1, trainable=False)

        @tf.function
        def trace_fn(loss, grad, vars):
            if tf.math.mod(step_num, 200) == 0:
                tf.print("[", step_num, "/", niter, "]  loss: ", loss, sep='')
            step_num.assign(step_num + 1)
            #  anneal temperature
            self.qF_temperature_var.assign(
                    self.init_temp * 0.5 ** tf.cast(step_num/niter, tf.float32))
            return loss

        trace = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn=lambda *args: model.log_prob(args),
            surrogate_posterior=variational_model,
            optimizer=tf.optimizers.Adam(learning_rate=1e-2),
            sample_size=1,
            num_steps=niter,
            trace_fn=trace_fn)

        class_probs = tf.nn.softmax(self.qF_logits_var, axis=1).numpy()
        return class_probs, self.qw_loc_var.numpy(), self.qz_loc_var.numpy()
