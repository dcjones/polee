
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
        if self.confounders is not None:
            F_full = tf.concat([F_full, tf.expand_dims(self.confounders, 0)], axis=-1)

        # tf.print("F_full", F_full, summarize=10000)

        return F_full

    def surrogate_design_matrix_model_fn(self, qF_logits_var, qF_temperature_var):
        qF_test = yield JDCRoot(Independent(tfd.RelaxedOneHotCategorical(
            temperature=qF_temperature_var,
            logits=qF_logits_var)))

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

        variational_model = tfd.JointDistributionCoroutine(
            lambda: self.variational_model_fn(
                lambda: self.surrogate_design_matrix_model_fn(
                    qF_logits_var, qF_temperature_var),
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
            optimizer=tf.optimizers.Adam(learning_rate=1e-3),
            sample_size=1,
            num_steps=niter,
            trace_fn=trace_fn)

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

        # @tf.function(autograph=False)
        # def grads(alter):
        #     with tf.GradientTape() as t:
        #         t.watch(qF_logits_var)

        #         # q_samples = partial_variational_model.sample(1)
        #         # loss = nest_util.call_fn(lambda *args: partial_model.log_prob(args), q_samples)
        #         loss = 0.0

        #         logits = qF_logits_var[0,]

        #         # if alter:
        #         #     a = np.zeros(logits.shape)
        #         #     a[0,0] = 5.0
        #         #     a[0,1] = -5.0
        #         #     a[0,2] = -5.0
        #         #     logits = logits + a

        #         F = tfd.Deterministic(
        #             loc=tf.nn.softmax(logits, axis=1)).sample()

        #         loss += tf.reduce_sum(tfd.OneHotCategorical(
        #             logits=tf.zeros([self.num_testing_samples, self.num_factors - self.num_confounders])).log_prob(F))

        #         x_scale = tf.nn.softplus(self.qx_scale_loc_var)

        #         x_loc = tf.matmul(F, self.qw_loc_var) + self.qx_bias_loc_var

        #         qx = tfd.Normal(
        #             loc=self.qx_loc_var[self.num_training_samples:,],
        #             scale=tf.nn.softplus(self.qx_softplus_scale_var[self.num_training_samples:,])).sample()

        #         loss += tf.reduce_sum(tfd.Normal(
        #             loc=x_loc - self.sample_scales[self.num_training_samples:],
        #             scale=x_scale).log_prob(qx))

        #     # return t.gradient(loss, qF_logits_var)
        #     return loss

        # print(grads(False).numpy()[0,])
        # print(grads(True).numpy()[0,])
        # print(grads(False).numpy())
        # print(grads(True).numpy())

        # print(qF_logits_var.numpy()[0,])

        #     tf.print("grads", tf.gradients(qF_logits_var, qF_logits_var)[0,])

        # TODO: How do I get to the bottom of this?
        #
        #

        # grads()

        print(qF_logits_var[0,].numpy())

        # I = tf.linalg.diag(tf.ones(self.num_factors))
        # # I = tf.constant(np.array([
        # #     [0.33, 0.33, 0.33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # #     [0.35, 0.32, 0.32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # #     [0.32, 0.35, 0.32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # #     [0.32, 0.32, 0.35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32))


        # # idx = self.qx_bias_loc_var > -12.0
        # idx = self.qx_bias_loc_var > -1000.0
        # print(np.sum(idx))

        # x_loc = tf.matmul(I, self.qw_loc_var.numpy()[:,idx]) + self.qx_bias_loc_var.numpy()[idx]
        # print(x_loc.numpy()[0:3,:])
        # # # scale = tf.nn.softplus(self.qx_scale_loc_var)
        # x_scale = tf.nn.softplus(self.qx_scale_loc_var.numpy()[idx])

        # qx = tfd.Normal(
        #     loc=self.qx_loc_var.numpy()[self.num_training_samples:self.num_training_samples+1,idx],
        #     scale=tf.nn.softplus(self.qx_softplus_scale_var.numpy()[self.num_training_samples:self.num_training_samples+1,idx])).sample()

        # # with open("debug-dump.csv", "w") as out:
        # #     out.write("x,loc1,loc2,loc3\n")
        # #     x = qx.numpy()
        # #     x_loc_ = x_loc.numpy()
        # #     for i in range(x.shape[1]):
        # #         out.write("{},{},{},{}\n".format(x[0,i], x_loc[0,i], x_loc[1,i], x_loc[2,i]))

        # print("MAX DIFF")
        # print(np.max(x_loc - qx, axis=1))
        # print(np.quantile(x_loc - qx, q=0.5, axis=1))
        # print(self.sample_scales[self.num_training_samples])
        # print(self.qw_distortion_c_loc_var)

        # lp = tfd.Normal(
        #     loc=x_loc - self.sample_scales[self.num_training_samples],
        #     scale=1.0).log_prob(qx).numpy()

        # # print(lp.shape)
        # print(np.sum(lp, axis=1))

        # print(tf.nn.softmax(qF_logits_var, axis=1).numpy()[0,])

        class_probs = tf.nn.softmax(qF_logits_var, axis=1).numpy()
        return class_probs

    def partial_model_fn(self, design_matrix_model, likelihood_model, sample_scales):
        # # Horseshoe prior on coefficients
        # # Weird parameterization is from tfp.sts.SparseLinearRegression This
        # # works because a standard cauchy is a t distribution with df=1, and a
        # # t-distribution can be expressed as a gamma-normal compound.

        # w_global_scale_variance = yield JDCRoot(Independent(tfd.InverseGamma(
        #     concentration=0.5, scale=0.5)))
        # w_global_scale_noncentered = yield JDCRoot(Independent(tfd.HalfNormal(
        #     scale=1.0)))
        # w_global_scale = w_global_scale_noncentered * tf.sqrt(w_global_scale_variance)

        # w_local_scale_variance = yield JDCRoot(Independent(tfd.InverseGamma(
        #     concentration=tf.fill([self.num_factors, self.num_features], 0.5),
        #     scale=tf.fill([self.num_factors, self.num_features], 0.5))))
        # w_local_scale_noncentered = yield JDCRoot(Independent(tfd.HalfNormal(
        #     scale=tf.ones([self.num_factors, self.num_features]))))
        # w_local_scale = w_local_scale_noncentered * tf.sqrt(w_local_scale_variance)

        # w = yield Independent(tfd.Normal(
        #     loc=tf.zeros([self.num_factors, self.num_features]),
        #     scale=w_global_scale * w_local_scale))

        # x_bias = yield JDCRoot(Independent(tfd.Normal(
        #     loc=tf.fill([self.num_features], np.float32(self.x_bias_loc0)),
        #     scale=np.float32(self.x_bias_scale0))))

        # weights = kernel_regression_weights(
        #     self.kernel_regression_bandwidth, x_bias, self.x_scale_hinges)
        # weights = kernel_regression_weights(
        #     self.kernel_regression_bandwidth, self.qx_bias_loc_var, self.x_scale_hinges)

        # # # Kernel regression correction.
        # # # This can be thought of as encoding the prior that similarly expression
        # # # genes are probably not all DE in the same direction. This can help
        # # # in situations where there are large differences in sequencing depth
        # # # that are highly correlated with factors in the regression. Differences
        # # # in sequencing depth alter the likelihood function of transcripts with
        # # # no reads which can otherwise manifest as  bias in the w
        # # # posterior.

        # w_distortion_c = yield Independent(tfd.Normal(
        #     loc=tf.zeros([self.num_factors, self.kernel_regression_degree]),
        #     scale=1.0))

        # w_distortion = tf.matmul(
        #     tf.expand_dims(w_distortion_c, 1),
        #     tf.expand_dims(weights, 0)) # [num_factors, num_features]

        F = yield from design_matrix_model()

        # the actual regression part
        # x_loc = tf.matmul(F, w + w_distortion) + x_bias
        # x_loc = tf.matmul(F, w) + x_bias
        x_loc = tf.matmul(F, self.qw_loc_var) + self.qx_bias_loc_var


        # Kernel regression mean-variance model
        # Biological variance is InverseGamma distributed with the mode
        # determined by kernel regression against the mean expression.

        # x_scale_concentration_c = yield JDCRoot(Independent(tfd.HalfCauchy(
        #     loc=tf.zeros([self.kernel_regression_degree]), scale=10.0)))

        # x_scale_scale_c = yield JDCRoot(Independent(tfd.HalfCauchy(
        #     loc=tf.zeros([self.kernel_regression_degree]), scale=10.0)))

        # x_scale = yield Independent(mean_variance_model(
        #     weights, x_scale_concentration_c, x_scale_scale_c))
        x_scale = tf.nn.softplus(self.qx_scale_loc_var)

        # x_inv_df = yield JDCRoot(Independent(tfd.HalfCauchy(loc=0.0, scale=1.0)))

        x = yield Independent(tfd.Normal(
            loc=x_loc - sample_scales[self.num_training_samples:],
            scale=x_scale))

        # if not self.use_point_estimates:
        #     # penalty for scale drift
        #     num_samples = len(sample_scales)
        #     # x_sample_scale = yield Independent(tfd.Normal(
        #     #     loc=tf.zeros(num_samples),
        #     #     scale=5e-4))

        # yield from likelihood_model(x)

    def partial_variational_model_fn(self, surrogate_design_matrix_model, surrogate_likelihood_model):
        # TODO: to run evaluation now we are going to have to swap out qx.
        # Can I get away with just replacing the variable in the class? I guess
        # we'll find out.

        # qw_global_scale_variance = yield JDCRoot(Independent(SoftplusNormal(
        #     loc=self.qw_global_scale_variance_loc_var,
        #     scale=tf.nn.softplus(self.qw_global_scale_variance_softplus_scale_var))))

        # qw_global_scale_noncentered = yield JDCRoot(Independent(SoftplusNormal(
        #     loc=self.qw_global_scale_noncentered_loc_var,
        #     scale=tf.nn.softplus(self.qw_global_scale_noncentered_softplus_scale_var))))

        # qw_local_scale_variance = yield JDCRoot(Independent(SoftplusNormal(
        #     loc=self.qw_local_scale_variance_loc_var,
        #     scale=tf.nn.softplus(self.qw_local_scale_variance_softplus_scale_var))))

        # qw_local_scale_noncentered = yield JDCRoot(Independent(SoftplusNormal(
        #     loc=self.qw_local_scale_noncentered_loc_var,
        #     scale=tf.nn.softplus(self.qw_local_scale_noncentered_softplus_scale_var))))

        # qw = yield JDCRoot(Independent(tfd.Normal(
        #     loc=self.qw_loc_var,
        #     scale=tf.nn.softplus(self.qw_softplus_scale_var))))

        # qx_bias = yield JDCRoot(Independent(tfd.Normal(
        #     loc=self.qx_bias_loc_var,
        #     scale=tf.nn.softplus(self.qx_bias_softplus_scale_var))))

        # qw_distortion_c = yield JDCRoot(Independent(tfd.Deterministic(
        #     loc=self.qw_distortion_c_loc_var)))

        yield from surrogate_design_matrix_model()

        # qx_scale_concentration_c = yield JDCRoot(Independent(tfd.Deterministic(
        #     loc=tf.nn.softplus(self.qx_scale_concentration_c_loc_var))))

        # qx_scale_scale_c = yield JDCRoot(Independent(tfd.Deterministic(
        #     loc=tf.nn.softplus(self.qx_scale_scale_c_loc_var))))

        # qx_scale = yield JDCRoot(Independent(SoftplusNormal(
        #     loc=self.qx_scale_loc_var,
        #     scale=tf.nn.softplus(self.qx_scale_softplus_scale_var))))

        # qx_inv_df = yield JDCRoot(Independent(tfd.Deterministic(
        #     loc=tf.nn.softplus(self.qx_inv_df_softplus_loc_var))))

        qx = yield JDCRoot(Independent(tfd.Normal(
            loc=self.qx_loc_var[self.num_training_samples:,],
            scale=tf.nn.softplus(self.qx_softplus_scale_var[self.num_training_samples:,]))))

            # qx_sample_scale = yield Independent(tfd.Deterministic(
            #     loc=tf.math.log(tf.reduce_sum(tf.math.exp(self.qx_loc_var), axis=-1))))

        # yield from surrogate_likelihood_model(qx)

    def partial_design_matrix_model_fn(self):
        F_test = yield JDCRoot(Independent(tfd.OneHotCategorical(
            logits=tf.zeros([self.num_testing_samples, self.num_factors]))))

        # F_full = tf.concat([tf.expand_dims(self.F, 0), F_test], axis=-2)
        # return F_full
        return F_test

    def partial_surrogate_design_matrix_model_fn(self, qF_logits_var, qF_temperature_var):
        # qF_test = yield JDCRoot(Independent(tfd.RelaxedOneHotCategorical(
        #     temperature=qF_temperature_var,
        #     logits=qF_logits_var)))
        qF_test = yield JDCRoot(Independent(tfd.Deterministic(
            loc=tf.nn.softmax(qF_logits_var, axis=1))))
