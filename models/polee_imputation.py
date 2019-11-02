
"""
Regression that inputes missing labels.
"""

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
from tensorflow_probability.python.internal import nest_util



import functools
from tensorflow_probability.python.vi import csiszar_divergence


class RNASeqImputedTranscriptLinearRegression(polee_regression.RNASeqTranscriptLinearRegression):
    # def __init__(
    #         self, vars, x_init, F_arr, num_confounders,
    #         sample_scales, use_point_estimates,
    #         kernel_regression_degree=15, kernel_regression_bandwidth=1.0):

    #     num_factors = F_arr.shape[1]
    #     self.num_confounders = num_confounders
    #     num_factors += num_confounders

    #     num_samples = len(sample_scales)
    #     self.num_training_samples = int(F_arr.shape[0])
    #     self.num_testing_samples = num_samples - self.num_training_samples

    #     super(RNASeqImputedTranscriptLinearRegression, self).__init__(
    #         vars, x_init, F_arr,
    #         sample_scales, use_point_estimates,
    #         kernel_regression_degree, kernel_regression_bandwidth,
    #         num_samples=num_samples, num_factors=num_factors)

    def __init__(
            self, vars, x_init, F_arr, confounders,
            sample_scales, use_point_estimates,
            kernel_regression_degree=15, kernel_regression_bandwidth=1.0):

        num_factors = F_arr.shape[1]
        if confounders is not None:
            num_factors += confounders.shape[1]
            self.num_confounders = int(confounders.shape[1])
        else:
            self.num_confounders = 0

        num_samples = len(sample_scales)
        self.num_training_samples = int(F_arr.shape[0])
        self.num_testing_samples = num_samples - self.num_training_samples
        self.confounders = confounders

        super(RNASeqImputedTranscriptLinearRegression, self).__init__(
            vars, x_init, F_arr,
            sample_scales, use_point_estimates,
            kernel_regression_degree, kernel_regression_bandwidth,
            num_samples=num_samples, num_factors=num_factors)

    def design_matrix_model_fn(self):
        F_test = yield JDCRoot(Independent(tfd.OneHotCategorical(
            logits=tf.zeros([
                self.num_testing_samples, self.num_factors - self.num_confounders]))))

        F_full = tf.concat([tf.expand_dims(self.F, 0), F_test], axis=-2)

        # F_confounders = yield JDCRoot(Independent(tfd.Normal(
        #     loc=tf.zeros([self.num_samples, self.num_confounders]),
        #     scale=1.0)))

        # F_full = tf.concat([F_full, F_confounders], axis=-1)

        if self.confounders is not None:
            F_full = tf.concat([F_full, tf.expand_dims(self.confounders, 0)], axis=-1)

        return F_full

    def surrogate_design_matrix_model_fn(
            self, qF_logits_var, qF_temperature_var,
            qF_confounders_loc_var, qF_confounders_softplus_scale_var):
        qF_test = yield JDCRoot(Independent(tfd.RelaxedOneHotCategorical(
            temperature=qF_temperature_var,
            logits=qF_logits_var)))

        # qF_confounders = yield JDCRoot(Independent(tfd.Normal(
        #     loc=qF_confounders_loc_var,
        #     scale=tf.nn.softplus(qF_confounders_softplus_scale_var))))

    def fit(self, niter):
        model = tfd.JointDistributionCoroutine(
            lambda: self.model_fn(
                self.design_matrix_model_fn,
                self.likelihood_model,
                self.sample_scales))

        qF_logits_var = tf.Variable(tf.zeros([
            self.num_testing_samples, self.num_factors - self.num_confounders]))
        init_temp = 5.0
        qF_temperature_var = tf.Variable(init_temp, trainable=False)

        qF_confounders_loc_var = tf.Variable(
            0.01 * tf.random.normal([self.num_samples, self.num_confounders]))
        qF_confounders_softplus_scale_var = tf.Variable(
            tf.fill([self.num_samples, self.num_confounders], 0.0))

        variational_model = tfd.JointDistributionCoroutine(
            lambda: self.variational_model_fn(
                lambda: self.surrogate_design_matrix_model_fn(
                    qF_logits_var, qF_temperature_var,
                    qF_confounders_loc_var, qF_confounders_softplus_scale_var),
                self.surrogate_likelihood_model))

        step_num = tf.Variable(1, trainable=False)

        @tf.function
        def trace_fn(loss, grad, vars):
            if tf.math.mod(step_num, 200) == 0:
                tf.print("[", step_num, "/", niter, "]  loss: ", loss, sep='')
            step_num.assign(step_num + 1)
            #  anneal temperature
            qF_temperature_var.assign(
                    init_temp * 0.5 ** tf.cast(step_num/niter, tf.float32))
            return loss

        trace = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn=lambda *args: model.log_prob(args),
            surrogate_posterior=variational_model,
            optimizer=tf.optimizers.Adam(learning_rate=1e-2),
            sample_size=1,
            num_steps=niter,
            trace_fn=trace_fn)

        # tf.print("qF_confounders_loc_var", qF_confounders_loc_var, summarize=100)

        partial_model = tfd.JointDistributionCoroutine(
            lambda: self.partial_model_fn(
                self.partial_design_matrix_model_fn,
                self.likelihood_model,
                self.sample_scales))

        partial_variational_model = tfd.JointDistributionCoroutine(
            lambda: self.partial_variational_model_fn(
                lambda: self.partial_surrogate_design_matrix_model_fn(
                    qF_logits_var, qF_temperature_var),
                self.surrogate_likelihood_model))

        factor_means = self.qx_bias_loc_var + self.qw_loc_var

        class_probs = tf.nn.softmax(qF_logits_var, axis=1).numpy()
        return class_probs, self.qw_loc_var.numpy(), factor_means.numpy(), tf.nn.softplus(self.qx_scale_loc_var).numpy(), self.qx_loc_var.numpy()

