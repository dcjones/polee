import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from polee_approx_likelihood import *
from polee_training import *
from polee_transcript_expression import *


"""
Fake distribution for evaluating gene expression given an approximate transcript
expression likelihood function.
"""
class RNASeqGeneApproxLikelihoodDist(tfp.distributions.Distribution):
    def __init__(self, vars, feature_idxs, transcript_idxs, feature_sizes,
            x_gene, x_isoform, name="RNASeqGeneApproxLikelihoodDist"):

        # expected shapes
        #   * x_gene: [1, num_samples, num_features]
        #   * x_isoform: [1, num_samples, n]

        n = np.max(transcript_idxs)
        num_features = np.max(feature_idxs)
        num_samples = int(x_gene.shape[1])

        # SparseTensor operations don't support batching. We get around that
        # here by flattening the input and using one big blockwise matrix.

        x_gene_flat = tf.reshape(x_gene, [num_features*num_samples, 1])
        x_gene_flat_exp = tf.math.exp(x_gene_flat)

        x_isoform_flat = tf.reshape(x_isoform, [n*num_samples])
        x_isoform_flat_exp = tf.math.exp(x_isoform_flat)

        x_isoform_indices = np.empty([n*num_samples, 2], dtype=np.int)
        k = 0
        for l in range(num_samples):
            for (i, j) in zip(transcript_idxs, feature_idxs):
                x_isoform_indices[k, 0] = (i - 1) + (l * n)
                x_isoform_indices[k, 1] = (j - 1) + (l * num_features)
                k += 1
        assert k == n*num_samples

        x_isoform_matrix_exp = tf.SparseTensor(
            indices=x_isoform_indices,
            values=x_isoform_flat_exp,
            dense_shape=[num_samples*n, num_samples*num_features])

        x_flat_normalizer = tf.sparse.sparse_dense_matmul(
                x_isoform_matrix_exp,
                tf.ones([n*num_samples, 1]),
                adjoint_a=True)

        x_flat_exp = tf.sparse.sparse_dense_matmul(
                x_isoform_matrix_exp,
                x_gene_flat_exp / x_flat_normalizer)
        x_flat = tf.math.log(x_flat_exp)

        x = tf.reshape(x_flat, [1, num_samples, n])

        self.x_gene = x_gene
        self.transcript_likelihood = rnaseq_approx_likelihood_from_vars(vars, x)
        self.feature_sizes = feature_sizes

        parameters = dict(locals())

        super(RNASeqGeneApproxLikelihoodDist, self).__init__(
              dtype=self.x_gene.dtype,
              reparameterization_type=tfp.distributions.FULLY_REPARAMETERIZED,
              validate_args=False,
              allow_nan_stats=False,
              parameters=parameters,
              name=name)

    def _sample_n(self, N, seed=None):
        shape = (N,) + self._batch_shape()
        return tf.zeros(shape)

    def _event_shape(self):
        return tf.TensorShape([])

    def _batch_shape(self):
        return self.x_gene.get_shape()[:-1]

    # @tf.function
    def _log_prob(self, __ignored__):
        return self.transcript_likelihood.log_prob(__ignored__) + \
            noninformative_gene_prior(self.x_gene, self.feature_sizes)


"""
Fake distribution for approximated feature likelihood.
"""
class RNASeqFeatureApproxLikelihoodDist(tfp.distributions.Distribution):
    def __init__(self, loc, scale, feature_sizes, x, name="RNASeqFeatureApproxLikelihoodDist"):
        self.loc = loc
        self.scale = scale
        self.feature_sizes = feature_sizes
        self.x = x

        parameters = dict(locals())

        super(RNASeqFeatureApproxLikelihoodDist, self).__init__(
              dtype=self.x.dtype,
              reparameterization_type=tfp.distributions.FULLY_REPARAMETERIZED,
              validate_args=False,
              allow_nan_stats=False,
              parameters=parameters,
              name=name)

    def _sample_n(self, N, seed=None):
        shape = (N,) + self._batch_shape()
        return tf.zeros(shape)

    def _event_shape(self):
        return tf.TensorShape([])

    def _batch_shape(self):
        return self.x.get_shape()[:-1]

    # @tf.function
    def _log_prob(self, __ignored__):
        x_gene = tf.math.log(tf.nn.softmax(self.x, axis=-1))

        feature_likelihood = tfp.distributions.Normal(
            loc=self.loc,
            scale=self.scale)

        return tf.reduce_sum(feature_likelihood.log_prob(x_gene), axis=-1)
            # noninformative_gene_prior(x_gene, self.feature_sizes)


"""
Compute gene expression from transcript expression (x).

`feature_idxs` and `transcript_idxs` are two vectors of equal length assigning
transcripts to features.
"""
def transcript_expression_to_feature_expression(
        num_features, n, feature_idxs, transcript_idxs, x):

    indices = np.transpose(np.vstack([feature_idxs, transcript_idxs]))

    feature_matrix = tf.SparseTensor(
        indices=indices,
        values=tf.ones(indices.shape[0], tf.float32),
        dense_shape=[num_features, n])

    return tf.transpose(tf.sparse.sparse_dense_matmul(
        feature_matrix, x, adjoint_b=True, name="x_feature"))


"""
Since likelihood is in terms of transcript expression, gene expression models
can have the undesirable property that there is effectively a uniform prior
over transcript expression, leading to a non-uniform prior over gene
expression. This corrects for that by computing the log determinant of the
Jacobian for the transcript expression to gene expression bijection.
"""
def noninformative_gene_prior(x_gene, gene_sizes):
    return tf.reduce_sum(-(gene_sizes-1) * x_gene, axis=-1)


# @tf.function
def sample_feature_expression(
        num_samples, num_features, n, vars, feature_idxs, transcript_idxs):
    x = rnaseq_approx_likelihood_sampler_from_vars(num_samples, n, vars)

    x_feature_exp = transcript_expression_to_feature_expression(
        num_features, n, feature_idxs, transcript_idxs, x)

    x_feature = tf.math.log(x_feature_exp)

    return x_feature


"""
Approximate likelihood function for features (typically gene), where a
"feature" is a set of transcripts. Approximated using a normal distribution
and minimizing KL(p||q), where `p` is the "true" distribution (not really
true, since it itself is Polya tree approximation).

This is very similar to what we do with splicing likelihood.
"""
def approximate_feature_likelihood(
        init_feed_dict, vars, num_samples, num_features, n,
        feature_idxs, transcript_idxs):

    num_mean_est_samples = 1000
    qx_feature_loc = tf.Variable(tf.zeros([num_samples, num_features]))

    for i in range(num_mean_est_samples):
        qx_feature_loc.assign_add(sample_feature_expression(
            num_samples, num_features, n, vars, feature_idxs, transcript_idxs))
    qx_feature_loc.assign(qx_feature_loc / num_mean_est_samples)

    num_var_est_samples = 1000
    qx_feature_scale = tf.Variable(tf.zeros([num_samples, num_features]))
    for i in range(num_mean_est_samples):
        qx_feature_scale.assign_add(tf.square(qx_feature_loc - sample_feature_expression(
            num_samples, num_features, n, vars, feature_idxs, transcript_idxs)))
    qx_feature_scale.assign(tf.sqrt(qx_feature_scale / num_var_est_samples))

    return (
        qx_feature_loc.numpy(),
        qx_feature_scale.numpy() )


"""
Use approximate feature expression likelihood to estimate posteriors.
"""
def estimate_feature_expression(
        init_feed_dict, vars, num_samples, num_features, n,
        feature_idxs, transcript_idxs, sess=None, sigma0=4.0):

    if sess is None:
        sess = tf.Session()

    x_likelihood_loc, x_likelihood_scale = approximate_feature_likelihood(
        init_feed_dict, vars, num_samples, num_features, n,
        feature_idxs, transcript_idxs, sess=sess)

    return estimate_feature_expression_from_normal_approx(
        init_feed_dict, vars, num_samples, num_features, n,
        x_likelihood_loc, x_likelihood_scale, sess=sess, sigma0=sigma0)

"""
If x_likelihood_loc and x_likelihood_scale are parameters to an approximated
likelihood function for x, estimate x using a pooled mean.
"""
def estimate_feature_expression_from_normal_approx(
        init_feed_dict, vars, num_samples, num_features, n,
        x_likelihood_loc, x_likelihood_scale, sess=None,
        softmax_x=True, mu0=None, sigma0=4.0):

    if sess is None:
        sess = tf.Session()

    # reusing transcript expression model, since this will be the same except for
    # handling of likelihood
    log_joint = ed.make_log_joint_fn(
        lambda: transcript_expression_model(
            num_samples, num_features, mu0=mu0, sigma0=sigma0))

    qx_mu_mu_param = tf.Variable(
        np.mean(x_likelihood_loc, 0),
        name="qx_mu_mu_param",
        dtype=tf.float32)
    qx_mu_softplus_sigma_param = tf.Variable(
        tf.fill([num_features], -1.0),
        name="qx_mu_softplus_sigma_param",
        dtype=tf.float32)

    qx_sigma_sq_mu_param = tf.Variable(
        tf.fill([num_features], 0.0),
        name="qx_sigma_sq_mu_param",
        dtype=tf.float32)
    qx_sigma_sq_softplus_sigma_param = tf.Variable(
        tf.fill([num_features], 1.0),
        name="qx_sigma_sq_softplus_sigma_param",
        dtype=tf.float32)

    qx_mu_param = tf.Variable(
        x_likelihood_loc,
        name="qx_mu_param",
        dtype=tf.float32)
    qx_softplus_sigma_param = tf.Variable(
        tf.fill([num_samples, num_features], -1.0),
        name="qx_softplus_sigma_param",
        dtype=tf.float32)

    qx_mu, qx_sigma_sq, qx = transcript_expression_variational_model(
        qx_mu_mu_param, qx_mu_softplus_sigma_param,
        qx_sigma_sq_mu_param, qx_sigma_sq_softplus_sigma_param,
        qx_mu_param, qx_softplus_sigma_param)

    lp = log_joint(
        x_mu=qx_mu,
        x_sigma_sq=qx_sigma_sq,
        x=qx)

    variational_log_joint = ed.make_log_joint_fn(
        lambda: transcript_expression_variational_model(
            qx_mu_mu_param, qx_mu_softplus_sigma_param,
            qx_sigma_sq_mu_param, qx_sigma_sq_softplus_sigma_param,
            qx_mu_param, qx_softplus_sigma_param))

    entropy = -variational_log_joint(
        qx_mu=qx_mu,
        qx_sigma_sq=qx_sigma_sq,
        qx=qx)

    approx_likelihood_dist = tfd.Normal(loc=x_likelihood_loc, scale=x_likelihood_scale)

    qx_ = tf.log(tf.nn.softmax(qx)) if softmax_x else qx
    approx_likelihood = tf.reduce_sum(approx_likelihood_dist.log_prob(qx_))

    elbo = lp + approx_likelihood + entropy

    if sess is None:
        sess = tf.Session()
    train(sess, -elbo, init_feed_dict, 500, 2e-2)

    return sess.run(qx.distribution.loc), sess.run(qx.distribution.scale)
