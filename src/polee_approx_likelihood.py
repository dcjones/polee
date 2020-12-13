
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python import framework
import tensorflow_probability as tfp
import numpy as np


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
Approximated RNA-Seq likelihood using a shared transformation.
"""
class RNASeqSharedPTTApproxLikelihoodDist(tfp.distributions.Distribution):
    def __init__(self, x, efflens,
                 la_mu,
                 la_sigma,
                 la_alpha,
                 left_index,
                 right_index,
                 leaf_index,
                 name="RNASeqSharedPTTApproxLikelihood"):
        self.x           = x
        self.efflens     = efflens
        self.mu          = la_mu
        self.sigma       = la_sigma
        self.alpha       = la_alpha

        parameters = dict(locals())

        num_nodes = left_index.shape[-1]
        n = (num_nodes+1)//2

        # Construct two sparse matrices that take cumsum(x) and compute
        # for each internal node i the intermediate values for its left and
        # right child, respectively.

        Lindexes = np.zeros([2*(n-1), 2], np.int)
        Lvalues = np.zeros([2*(n-1)], np.float64)

        Rindexes = np.zeros([2*(n-1), 2], np.int)
        Rvalues = np.zeros([2*(n-1)], np.float64)

        Uindexes = np.zeros([2*(n-1), 2], np.int)
        Uvalues = np.zeros([2*(n-1)], np.float64)

        # for each internal node we need to know its min and max leaf child
        min_leaf_index = np.zeros([num_nodes], np.int)
        max_leaf_index = np.zeros([num_nodes], np.int)

        # set values for leaf nodes
        k = 0
        for i in range(num_nodes-1, -1, -1):
            if leaf_index[0, i] >= 0:
                min_leaf_index[i] = k
                max_leaf_index[i] = k
                k += 1
            else:
                min_leaf_index[i] = min_leaf_index[left_index[0, i]]
                max_leaf_index[i] = max_leaf_index[right_index[0, i]]
                assert min_leaf_index[i] < max_leaf_index[i]

        # build sparse matrix for internal nodes
        k = 0 # internal node number
        for i in range(num_nodes):
            if leaf_index[0, i] >= 0:
                continue

            Uindexes[2*k+1, 0] = k
            Uindexes[2*k+1, 1] = max_leaf_index[i]
            Uvalues[2*k+1] = 1.0

            if min_leaf_index[i] > 0:
                Uindexes[2*k, 0] = k
                Uindexes[2*k, 1] = min_leaf_index[i]-1
                Uvalues[2*k] = -1.0
            else:
                # we don't need this entry, easier to just set it do arbitrary
                # index and zero than to try to remove it
                Uindexes[2*k, 0] = k
                if max_leaf_index[i] < n-1:
                    Uindexes[2*k, 1] = max_leaf_index[i] + 1
                else:
                    Uindexes[2*k, 1] = max_leaf_index[i] - 1
                Uvalues[2*k] = 0.0

            l = left_index[0, i]
            r = right_index[0, i]

            Lindexes[2*k+1, 0] = k
            Lindexes[2*k+1, 1] = max_leaf_index[l]
            Lvalues[2*k+1] = 1.0

            if min_leaf_index[l] > 0:
                Lindexes[2*k, 0] = k
                Lindexes[2*k, 1] = min_leaf_index[l]-1
                Lvalues[2*k] = -1.0
            else:
                # we don't need this entry, easier to just set it do arbitrary
                # index and zero than to try to remove it
                Lindexes[2*k, 0] = k
                if max_leaf_index[l] < n-1:
                    Lindexes[2*k, 1] = max_leaf_index[l] + 1
                else:
                    Lindexes[2*k, 1] = max_leaf_index[l] - 1
                Lvalues[2*k] = 0.0

            Rindexes[2*k+1, 0] = k
            Rindexes[2*k+1, 1] = max_leaf_index[r]
            Rvalues[2*k+1] = 1.0

            if min_leaf_index[r] > 0:
                Rindexes[2*k, 0] = k
                Rindexes[2*k, 1] = min_leaf_index[r]-1
                Rvalues[2*k] = -1.0
            else:
                # we don't need this entry, easier to just set it do arbitrary
                # index and zero than to try to remove it
                Rindexes[2*k, 0] = k
                if max_leaf_index[r] < n-1:
                    Rindexes[2*k, 1] = max_leaf_index[r] + 1
                else:
                    Rindexes[2*k, 1] = max_leaf_index[r] - 1
                Rvalues[2*k] = 0.0

            k += 1

        self.Linternal = tf.sparse.reorder(tf.sparse.SparseTensor(
            Lindexes, Lvalues, [n-1, n]))
        self.Rinternal = tf.sparse.reorder(tf.sparse.SparseTensor(
            Rindexes, Rvalues, [n-1, n]))
        self.Uinternal = tf.sparse.reorder(tf.sparse.SparseTensor(
            Uindexes, Uvalues, [n-1, n]))

        # Construct a permutation vector to reorder `x` into their leaf node
        # position.

        leaf_permutation = np.zeros(n, np.int)
        k = 0
        for i in range(num_nodes):
            if leaf_index[0, i] >= 0:
                leaf_permutation[k] = leaf_index[0, i]
                k += 1

        # node in the ptt are enumerate from right to left, insanely, so we have
        # to account for that.
        leaf_permutation = list(reversed(leaf_permutation))
        self.leaf_permutation = tfp.bijectors.Permute(permutation=leaf_permutation)

        super(RNASeqSharedPTTApproxLikelihoodDist, self).__init__(
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

        # compute softmax of x
        x = x_exp / tf.reduce_sum(x_exp, axis=-1, keepdims=True)

        # jacobian for softmax (softmax is not a bijection, so this is not imprecise)
        ladj -= (n-1) * tf.math.log(tf.reduce_sum(x_exp, axis=-1))

        # effective length transform
        # --------------------------

        x_scaled = x * self.efflens
        x_scaled_sum = tf.reduce_sum(x_scaled, axis=-1, keepdims=True)
        x_efflen = x_scaled / x_scaled_sum

        ladj += tf.expand_dims(tf.reduce_sum(tf.math.log(self.efflens), axis=-1), axis=0) - \
                tf.reshape(tf.math.log(x_scaled_sum), [1, num_samples])

        # Inverse ptt
        # -----------

        x_leaf = self.leaf_permutation.forward(x_efflen)
        C = tf.math.cumsum(tf.cast(x_leaf, tf.float64), axis=-1)

        # compute internal intermediate values
        Csqueeze = tf.squeeze(C, axis=0)

        u_left = tf.sparse.sparse_dense_matmul(
            self.Linternal, Csqueeze, adjoint_b=True)
        u_right = tf.sparse.sparse_dense_matmul(
            self.Rinternal, Csqueeze, adjoint_b=True)

        u = tf.sparse.sparse_dense_matmul(
            self.Uinternal, Csqueeze, adjoint_b=True)

        # TODO: This jacobian could be much for efficiently computed as
        # the sum leaf values times the lengths of the paths to the root
        ladj += -tf.reduce_sum(tf.math.log(tf.cast(u, tf.float32)), axis=0)

        y_logit = tf.transpose(
            tf.math.log(tf.cast(u_left, tf.float32)) -
            tf.math.log(tf.cast(u_right, tf.float32)))

        # ladj for the implicit logit transform we just did
        ladj += tf.reduce_sum(
            tf.math.log(2*(tf.math.cosh(y_logit) + 1)),
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

        # tf.print("scales", tf.reduce_sum(x_exp, axis=-1))

        # jacobian for exp transformation
        ladj += tf.reduce_sum(self.x, axis=-1)

        # compute softmax of x
        x = x_exp / tf.reduce_sum(x_exp, axis=-1, keepdims=True)

        # jacobian for softmax
        ladj -= (n-1) * tf.math.log(tf.reduce_sum(x_exp, axis=-1))

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
    efflens     = vars["efflen"]
    la_mu       = vars["la_mu"]
    la_sigma    = vars["la_sigma"]
    la_alpha    = vars["la_alpha"]
    left_index  = vars["left_index"]
    right_index = vars["right_index"]
    leaf_index  = vars["leaf_index"]

    # using a shared ptt
    if type(left_index) == np.ndarray:
        return RNASeqSharedPTTApproxLikelihoodDist(
            x=x,
            efflens=efflens,
            la_mu=la_mu,
            la_sigma=la_sigma,
            la_alpha=la_alpha,
            left_index=left_index,
            right_index=right_index,
            leaf_index=leaf_index)
    else:
        return RNASeqApproxLikelihoodDist(
            x=x,
            efflens=efflens,
            la_mu=la_mu,
            la_sigma=la_sigma,
            la_alpha=la_alpha,
            left_index=left_index,
            right_index=right_index,
            leaf_index=leaf_index)
