
"""
Bayesian linear regression.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import sys
import datetime
import polee
from polee_approx_likelihood import *
from polee_gene_expression import *
from polee import *


class RNASeqLinearRegression:
    def __init__(self, F, x_init, likelihood_model, surrogate_likelihood_model,
            x_bias_loc0, x_bias_scale0, x_scale_hinges, sample_scales,
            use_point_estimates, kernel_regression_degree, kernel_regression_bandwidth):
        self.num_samples = int(F.shape[0])
        self.num_factors = int(F.shape[1])
        self.num_features = int(x_init.shape[1])

        # TODO: I think my plan here is to pass in
        # likelihood_model and surrogate_likelihood_model, then create
        # sub-classes for transcript regression and so on.

        self.F = F
        self.likelihood_model = likelihood_model
        self.surrogate_likelihood_model = surrogate_likelihood_model
        self.x_bias_loc0 = x_bias_loc0
        self.x_bias_scale0 = x_bias_scale0
        self.x_scale_hinges = x_scale_hinges
        self.sample_scales = sample_scales
        self.use_point_estimates = use_point_estimates
        self.kernel_regression_degree = kernel_regression_degree
        self.kernel_regression_bandwidth = kernel_regression_bandwidth

        # Parameters for surrogate posterior
        self.qw_global_scale_variance_loc_var = tf.Variable(0.0)
        self.qw_global_scale_variance_softplus_scale_var = tf.Variable(-1.0)

        self.qw_global_scale_noncentered_loc_var = tf.Variable(0.0)
        self.qw_global_scale_noncentered_softplus_scale_var = tf.Variable(-1.0)

        self.qw_local_scale_variance_loc_var = tf.Variable(
            tf.fill([self.num_factors, self.num_features], 0.0))
        self.qw_local_scale_variance_softplus_scale_var = tf.Variable(
            tf.fill([self.num_factors, self.num_features], -1.0))

        self.qw_local_scale_noncentered_loc_var = tf.Variable(
            tf.fill([self.num_factors, self.num_features], 0.0))
        self.qw_local_scale_noncentered_softplus_scale_var = tf.Variable(
            tf.fill([self.num_factors, self.num_features], -1.0))

        self.qw_loc_var = tf.Variable(
            tf.zeros([self.num_factors, self.num_features]))
        self.qw_softplus_scale_var = tf.Variable(
            tf.fill([self.num_factors, self.num_features], -2.0))

        self.qx_bias_loc_var = tf.Variable(
            tf.reduce_mean(x_init, axis=0))
        self.qx_bias_softplus_scale_var = tf.Variable(
            tf.fill([self.num_features], -1.0))

        self.qw_distortion_c_loc_var = tf.Variable(
            tf.zeros([self.num_factors, self.kernel_regression_degree]))

        self.qx_scale_concentration_c_loc_var = tf.Variable(
            tf.fill([self.kernel_regression_degree], 1.0))

        self.qx_scale_scale_c_loc_var = tf.Variable(
            tf.fill([self.kernel_regression_degree], 1.0))

        self.qx_scale_loc_var = tf.Variable(
            tf.fill([self.num_features], -0.5))
        self.qx_scale_softplus_scale_var = tf.Variable(
            tf.fill([self.num_features], -1.0))

        self.qx_loc_var = tf.Variable(
            x_init,
            trainable=not use_point_estimates)
        self.qx_softplus_scale_var = tf.Variable(
            tf.fill([self.num_samples, self.num_features], -1.0))

    def model_fn(self, design_matrix_model, likelihood_model, sample_scales):
        # Horseshoe prior on coefficients
        # Weird parameterization is from tfp.sts.SparseLinearRegression This
        # works because a standard cauchy is a t distribution with df=1, and a
        # t-distribution can be expressed as a gamma-normal compound.

        w_global_scale_variance = yield JDCRoot(Independent(tfd.InverseGamma(
            concentration=0.5, scale=0.5)))
        w_global_scale_noncentered = yield JDCRoot(Independent(tfd.HalfNormal(
            scale=1.0)))
        w_global_scale = w_global_scale_noncentered * tf.sqrt(w_global_scale_variance)

        w_local_scale_variance = yield JDCRoot(Independent(tfd.InverseGamma(
            concentration=tf.fill([self.num_factors, self.num_features], 0.5),
            scale=tf.fill([self.num_factors, self.num_features], 0.5))))
        w_local_scale_noncentered = yield JDCRoot(Independent(tfd.HalfNormal(
            scale=tf.ones([self.num_factors, self.num_features]))))
        w_local_scale = w_local_scale_noncentered * tf.sqrt(w_local_scale_variance)

        w = yield Independent(tfd.Normal(
            loc=tf.zeros([self.num_factors, self.num_features]),
            scale=w_global_scale * w_local_scale))

        x_bias = yield JDCRoot(Independent(tfd.Normal(
            loc=tf.fill([self.num_features], np.float32(self.x_bias_loc0)),
            scale=np.float32(self.x_bias_scale0))))

        weights = kernel_regression_weights(
            self.kernel_regression_bandwidth, x_bias, self.x_scale_hinges)

        # Kernel regression correction.
        # This can be thought of as encoding the prior that similarly expression
        # genes are probably not all DE in the same direction. This can help
        # in situations where there are large differences in sequencing depth
        # that are highly correlated with factors in the regression. Differences
        # in sequencing depth alter the likelihood function of transcripts with
        # no reads which can otherwise manifest as  bias in the w
        # posterior.

        w_distortion_c = yield Independent(tfd.Normal(
            loc=tf.zeros([self.num_factors, self.kernel_regression_degree]),
            scale=1.0))

        w_distortion = tf.matmul(
            tf.expand_dims(w_distortion_c, 1),
            tf.expand_dims(weights, 0)) # [num_factors, num_features]

        F = yield from design_matrix_model()

        # the actual regression part
        x_loc = tf.matmul(F, w + w_distortion) + x_bias

        # Kernel regression mean-variance model
        # Biological variance is InverseGamma distributed with the mode
        # determined by kernel regression against the mean expression.

        x_scale_concentration_c = yield JDCRoot(Independent(tfd.HalfCauchy(
            loc=tf.zeros([self.kernel_regression_degree]), scale=10.0)))

        x_scale_scale_c = yield JDCRoot(Independent(tfd.HalfCauchy(
            loc=tf.zeros([self.kernel_regression_degree]), scale=10.0)))

        x_scale = yield Independent(mean_variance_model(
            weights, x_scale_concentration_c, x_scale_scale_c))

        x = yield Independent(tfd.Normal(
            loc=x_loc - sample_scales,
            scale=x_scale))

        if not self.use_point_estimates:
            # penalty for scale drift
            num_samples = len(sample_scales)
            x_sample_scale = yield Independent(tfd.Normal(
                loc=tf.zeros(num_samples),
                scale=5e-4))

        yield from likelihood_model(x)

    def variational_model_fn(self, surrogate_design_matrix_model, surrogate_likelihood_model):
        # TODO: to run evaluation now we are going to have to swap out qx.
        # Can I get away with just replacing the variable in the class? I guess
        # we'll find out.

        qw_global_scale_variance = yield JDCRoot(Independent(SoftplusNormal(
            loc=self.qw_global_scale_variance_loc_var,
            scale=tf.nn.softplus(self.qw_global_scale_variance_softplus_scale_var))))

        qw_global_scale_noncentered = yield JDCRoot(Independent(SoftplusNormal(
            loc=self.qw_global_scale_noncentered_loc_var,
            scale=tf.nn.softplus(self.qw_global_scale_noncentered_softplus_scale_var))))

        qw_local_scale_variance = yield JDCRoot(Independent(SoftplusNormal(
            loc=self.qw_local_scale_variance_loc_var,
            scale=tf.nn.softplus(self.qw_local_scale_variance_softplus_scale_var))))

        qw_local_scale_noncentered = yield JDCRoot(Independent(SoftplusNormal(
            loc=self.qw_local_scale_noncentered_loc_var,
            scale=tf.nn.softplus(self.qw_local_scale_noncentered_softplus_scale_var))))

        qw = yield JDCRoot(Independent(tfd.Normal(
            loc=self.qw_loc_var,
            scale=tf.nn.softplus(self.qw_softplus_scale_var))))

        qx_bias = yield JDCRoot(Independent(tfd.Normal(
            loc=self.qx_bias_loc_var,
            scale=tf.nn.softplus(self.qx_bias_softplus_scale_var))))

        qw_distortion_c = yield JDCRoot(Independent(tfd.Deterministic(
            loc=self.qw_distortion_c_loc_var)))

        yield from surrogate_design_matrix_model()

        qx_scale_concentration_c = yield JDCRoot(Independent(tfd.Deterministic(
            loc=tf.nn.softplus(self.qx_scale_concentration_c_loc_var))))

        qx_scale_scale_c = yield JDCRoot(Independent(tfd.Deterministic(
            loc=tf.nn.softplus(self.qx_scale_scale_c_loc_var))))

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

            qx_sample_scale = yield Independent(tfd.Deterministic(
                loc=tf.math.log(tf.reduce_sum(tf.math.exp(self.qx_loc_var), axis=-1))))

        yield from surrogate_likelihood_model(qx)

    def fit(self, niter):
        def design_matrix_model():
            return self.F
            yield

        model = tfd.JointDistributionCoroutine(
            lambda: self.model_fn(
                design_matrix_model,
                self.likelihood_model,
                self.sample_scales))

        def surrogate_design_matrix_model():
            return
            yield

        variational_model = tfd.JointDistributionCoroutine(
            lambda: self.variational_model_fn(
                surrogate_design_matrix_model,
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

        return (
            self.qx_loc_var.numpy(),
            self.qw_loc_var.numpy(),
            tf.nn.softplus(self.qw_softplus_scale_var).numpy(),
            self.qx_bias_loc_var.numpy(),
            tf.nn.softplus(self.qx_scale_loc_var).numpy())

    def classify(self, x_init, likelihood_model, surrogate_likelihood_model,
            sample_scales, use_point_estimates, niter, extra_training_vars=[]):

        num_testing_samples = len(sample_scales)

        def design_matrix_model():
            F = yield JDCRoot(Independent(tfd.OneHotCategorical(
                logits=tf.zeros([num_testing_samples, self.num_factors]))))
            return F

        model = tfd.JointDistributionCoroutine(
            lambda: self.model_fn(
                design_matrix_model,
                likelihood_model,
                sample_scales))

        qF_logits_var = tf.Variable(tf.zeros([num_testing_samples, self.num_factors]))
        init_temp = 5.0
        qF_temperature_var = tf.Variable(init_temp)

        qx_testing_loc_var = tf.Variable(x_init)
        qx_testing_softplus_scale_var = tf.Variable(
            tf.fill([num_testing_samples, self.num_features], -1.0))

        # swap out testing qx vars
        qx_training_loc_var = self.qx_loc_var
        qx_training_softplus_scale_var = self.qx_softplus_scale_var
        self.qx_loc_var = qx_testing_loc_var
        self.qx_softplus_scale_var = qx_testing_softplus_scale_var

        def surrogate_design_matrix_model():
            qF = yield JDCRoot(Independent(tfd.RelaxedOneHotCategorical(
                temperature=qF_temperature_var,
                logits=qF_logits_var)))

        variational_model = tfd.JointDistributionCoroutine(
            lambda: self.variational_model_fn(
                surrogate_design_matrix_model,
                surrogate_likelihood_model))

        step_num = tf.Variable(1, trainable=False)

        @tf.function
        def trace_fn(loss, grad, vars):
            if tf.math.mod(step_num, 200) == 0:
                tf.print("[", step_num, "/", niter, "]  loss: ", loss, sep='')
            step_num.assign(step_num + 1)
            #  anneal temperature
            qF_temperature_var.assign(
                    init_temp * 0.05 ** tf.cast(step_num/niter, tf.float32))
            return loss

        trainable_variables = [qF_logits_var]
        if not use_point_estimates:
            trainable_variables.append(qx_testing_loc_var)
            trainable_variables.append(qx_testing_softplus_scale_var)
        trainable_variables.extend(extra_training_vars)

        trace = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn=lambda *args: model.log_prob(args),
            surrogate_posterior=variational_model,
            optimizer=tf.optimizers.Adam(learning_rate=1e-3),
            sample_size=1,
            num_steps=niter,
            trace_fn=trace_fn,
            trainable_variables=trainable_variables)

        class_probs = tf.nn.softmax(qF_logits_var, axis=1).numpy()

        # swap back qx vars
        self.qx_loc_var = qx_training_loc_var
        self.qx_softplus_scale_var = qx_training_softplus_scale_var

        return class_probs



"""
Variational inference on transcript expression linear regression.
"""
class RNASeqTranscriptLinearRegression(RNASeqLinearRegression):
    def __init__(
            self, vars, x_init, F_arr,
            sample_scales, use_point_estimates,
            kernel_regression_degree=15, kernel_regression_bandwidth=1.0):

        F = tf.constant(F_arr, dtype=tf.float32)
        num_samples = x_init.shape[0]
        num_features = x_init.shape[1]

        x_init_mean = np.mean(x_init, axis=0)
        x_scale_hinges = tf.constant(
            choose_knots(np.min(x_init_mean), np.max(x_init_mean), kernel_regression_degree),
            dtype=tf.float32)

        x_bias_mu0 = np.log(1/num_features)
        x_bias_sigma0 = 12.0

        def likelihood_model(x):
            if not use_point_estimates:
                likelihood = yield tfd.Independent(rnaseq_approx_likelihood_from_vars(vars, x))

        def surrogate_likelihood_model(qx):
            if not use_point_estimates:
                qrnaseq_reads = yield JDCRoot(Independent(tfd.Deterministic(tf.zeros([num_samples]))))

        super(RNASeqTranscriptLinearRegression, self).__init__(
            F, x_init, likelihood_model, surrogate_likelihood_model,
            x_bias_mu0, x_bias_sigma0, x_scale_hinges, sample_scales,
            use_point_estimates,
            kernel_regression_degree, kernel_regression_bandwidth)

    def classify(self, vars, x_init, sample_scales, use_point_estimates, niter):
        x_init_mean = np.mean(x_init, axis=0)

        num_samples = len(sample_scales)

        def likelihood_model(x):
            if not use_point_estimates:
                likelihood = yield tfd.Independent(rnaseq_approx_likelihood_from_vars(vars, x))

        def surrogate_likelihood_model(qx):
            if not use_point_estimates:
                qrnaseq_reads = yield JDCRoot(Independent(tfd.Deterministic(tf.zeros([num_samples]))))

        return super(RNASeqTranscriptLinearRegression, self).classify(
            x_init, likelihood_model, surrogate_likelihood_model,
            sample_scales, use_point_estimates, niter)


"""
Variational inference on feature (i.e. gene) log-ratio linear regression.
"""
class RNASeqGeneLinearRegression(RNASeqLinearRegression):
    def __init__(
        self, vars,
        feature_idxs, transcript_idxs,
        x_gene_init, x_isoform_init,
        feature_sizes,
        F_arr, sample_scales, use_point_estimates,
        kernel_regression_degree=15, kernel_regression_bandwidth=1.0):

        num_samples = x_gene_init.shape[0]
        num_features = x_gene_init.shape[1]
        n = np.max(transcript_idxs)

        F = tf.constant(F_arr, dtype=tf.float32)

        x_gene_init_mean = np.mean(x_gene_init, axis=0)
        x_gene_scale_hinges = tf.constant(
            choose_knots(
                np.min(x_gene_init_mean), np.max(x_gene_init_mean), kernel_regression_degree),
            dtype=tf.float32)

        x_gene_bias_mu0 = np.log(1./num_features)
        x_gene_bias_sigma0 = 12.0

        def likelihood_model(x_gene):
            x_isoform_mean = yield JDCRoot(Independent(tfd.Normal(
                loc=tf.zeros([1,n]),
                scale=2.0)))

            x_isoform = yield JDCRoot(Independent(tfd.Normal(
                loc=x_isoform_mean,
                scale=tf.fill([num_samples, n], 1.0))))

            if not use_point_estimates:
                likelihood = yield tfd.Independent(RNASeqGeneApproxLikelihoodDist(
                    vars, feature_idxs, transcript_idxs, feature_sizes, x_gene, x_isoform))

        self.qx_isoform_mean_loc_var = tf.Variable(np.mean(x_isoform_init, axis=0, keepdims=True))
        self.qx_isoform_mean_softplus_scale_var = tf.Variable(tf.fill([1, n], -2.0))

        qx_isoform_loc_var = tf.Variable(x_isoform_init, trainable=not use_point_estimates)
        qx_isoform_softplus_scale_var = tf.Variable(tf.fill([num_samples, n], -2.0))

        def surrogate_likelihood_model(qx_gene):
            qx_isoform_mean = yield JDCRoot(Independent(
                tfd.Normal(
                    loc=self.qx_isoform_mean_loc_var,
                    scale=tf.nn.softplus(self.qx_isoform_mean_softplus_scale_var))))

            if use_point_estimates:
                qx_isoform = yield JDCRoot(Independent(
                    tfd.Deterministic(loc=qx_isoform_loc_var)))
            else:
                qx_isoform = yield JDCRoot(Independent(
                    tfd.Normal(
                        loc=qx_isoform_loc_var,
                        scale=tf.nn.softplus(qx_isoform_softplus_scale_var))))

                qrnaseq_reads = yield JDCRoot(Independent(tfd.Deterministic(tf.zeros([num_samples]))))

        super(RNASeqGeneLinearRegression, self).__init__(
            F, x_gene_init, likelihood_model, surrogate_likelihood_model,
            x_gene_bias_mu0, x_gene_bias_sigma0, x_gene_scale_hinges, sample_scales,
            use_point_estimates, kernel_regression_degree, kernel_regression_bandwidth)

    def classify(
            self, vars, feature_idxs, transcript_idxs,
            x_gene_init, x_isoform_init, feature_sizes,
            sample_scales, use_point_estimates, niter):

        num_samples = x_gene_init.shape[0]
        num_features = x_gene_init.shape[1]
        n = np.max(transcript_idxs)

        def likelihood_model(x_gene):
            x_isoform_mean = yield JDCRoot(Independent(tfd.Normal(
                loc=tf.zeros([1,n]),
                scale=2.0)))

            x_isoform = yield JDCRoot(Independent(tfd.Normal(
                loc=x_isoform_mean,
                scale=tf.fill([num_samples, n], 1.0))))

            if not use_point_estimates:
                likelihood = yield tfd.Independent(RNASeqGeneApproxLikelihoodDist(
                    vars, feature_idxs, transcript_idxs, feature_sizes, x_gene, x_isoform))

        qx_isoform_loc_var = tf.Variable(x_isoform_init)
        qx_isoform_softplus_scale_var = tf.Variable(tf.fill([num_samples, n], -2.0))

        def surrogate_likelihood_model(qx_gene):
            qx_isoform_mean = yield JDCRoot(Independent(
                tfd.Normal(
                    loc=self.qx_isoform_mean_loc_var,
                    scale=tf.nn.softplus(self.qx_isoform_mean_softplus_scale_var))))

            if use_point_estimates:
                qx_isoform = yield JDCRoot(Independent(
                    tfd.Deterministic(loc=qx_isoform_loc_var)))
            else:
                qx_isoform = yield JDCRoot(Independent(
                    tfd.Normal(
                        loc=qx_isoform_loc_var,
                        scale=tf.nn.softplus(qx_isoform_softplus_scale_var))))

                qrnaseq_reads = yield JDCRoot(Independent(tfd.Deterministic(tf.zeros([num_samples]))))

        extra_training_vars = [] if use_point_estimates else [qx_isoform_loc_var, qx_isoform_softplus_scale_var]

        predictions = super(RNASeqGeneLinearRegression, self).classify(
            x_gene_init, likelihood_model, surrogate_likelihood_model,
            sample_scales, use_point_estimates, niter,
            extra_training_vars=extra_training_vars)

        return predictions

