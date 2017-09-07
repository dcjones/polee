
import numpy as np
import tensorflow as tf
from tensorflow.contrib import distributions
from tensorflow.contrib import framework
import edward
from queue import Queue
import sys


class RNASeqApproxLikelihoodDist(distributions.Distribution):
    def __init__(self, x, efflens, As, node_parent_idxs, node_js,
                 validate_args=False,
                 allow_nan_stats=False,
                 name="RNASeqApproxLikelihood"):

        with tf.name_scope(name, values=[x]) as ns:
            self.x = tf.identity(x, name="rnaseq/x")
            framework.assert_same_float_dtype([self.x])
        parameters = locals()

        # print(self.x)
        # print(self.x.get_shape())

        # self.x = x
        self.efflens = efflens
        self.As = As
        self.node_parent_idxs = node_parent_idxs
        self.node_js = node_js

        super(RNASeqApproxLikelihoodDist, self).__init__(
              dtype=self.x.dtype,
              validate_args=validate_args,
              allow_nan_stats=allow_nan_stats,
              reparameterization_type=tf.contrib.distributions.FULLY_REPARAMETERIZED,
              parameters=parameters,
              graph_parents=[self.x,])

    def _get_event_shape(self):
        print("y.get_shape()")
        print(self.x.get_shape())
        return tf.TensorShape([2, self.x.get_shape()[-1] - 1])

    def _get_batch_shape(self):
        print("y.get_shape()")
        print(self.x.get_shape())
        return self.x.get_shape()[:-1]

    def _log_prob(self, musigma):

        # self.x = tf.Print(self.x, [tf.reduce_min(self.x[0,:]), tf.reduce_max(self.x[0,:])], "X[0,:] SPAN", summarize=10)
        self.x = tf.Print(self.x, [tf.reduce_sum(tf.exp(self.x), axis=1)], "SCALE", summarize=10)

        n = int(self.x.get_shape()[-1])

        num_samples = len(self.As)
        num_nodes = self.node_js.shape[0]
        print(num_nodes)

        y_tensors = []

        # TODO: This shit makes me real uncomfortable. This is not a bijection,
        # and there is no jacobian term. Suggests we are doing things wrong. I
        # could do something like exp(x_i) / (1 + sum(x_i)) which should be a
        # bijection with a well defined jacobian.
        # self_x = tf.Print(self.x, [tf.reduce_min(self.x), tf.reduce_max(self.x)], "X SPAN")
        x = tf.nn.softmax(self.x)

        # TODO: Let's see if this is sufficient adjustment
        x_softmax_ladj = tf.reduce_sum(self.x, axis=1)
        x_softmax_ladj = tf.Print(x_softmax_ladj, [tf.reduce_min(x_softmax_ladj), tf.reduce_max(x_softmax_ladj)], "SOFTMAX LADJ SPAN")

        x = tf.Print(x, [tf.reduce_min(x[0,:]), tf.reduce_max(x[0,:])], "SFTMX X[0,:] SPAN", summarize=10)
        # idx = 198973 - 1
        idx = 194630 - 1
        # x = tf.Print(x, [tf.reduce_min(x[:,idx]), tf.reduce_max(x[:,idx])], "X[:,IDX] SPAN", summarize=10)
        x = tf.Print(x, [x[:,idx]], "X[:,IDX]", summarize=10)

        # x = tf.Print(x, [tf.reduce_min(x), tf.reduce_max(x)], "SOFTMAX X SPAN")
        # x = tf.Print(x, [tf.reduce_sum(tf.cast(tf.logical_not(tf.is_finite(x)), tf.float32))],
        #              "SOFTMAX X NON-FINITE COUNT")

        # effective length transform
        x_scaled = tf.multiply(x, self.efflens)
        # x_scaled = tf.Print(x_scaled, [tf.reduce_min(x_scaled), tf.reduce_max(x_scaled)], "X_SCALED SPAN")
        # x_scaled = tf.Print(x_scaled, [tf.reduce_sum(tf.cast(tf.logical_not(tf.is_finite(x_scaled)), tf.float32))],
        #                     "X_SCALED NON-FINITE COUNT")
        x_scaled_sum = tf.reduce_sum(x_scaled, axis=1, keep_dims=True)
        # x_scaled_sum = tf.Print(x_scaled_sum, [x_scaled_sum], "X_SCALED_SUM", summarize=10)
        x_efflen = tf.divide(x_scaled, x_scaled_sum)
        # x_efflen = tf.Print(x_efflen, [tf.reduce_min(x_efflen[:,122775]), tf.reduce_max(x_efflen[:,122775])],"X_EFFLEN[:,IDX] SPAN", summarize=10)
        # x_efflen = x
        # x_scaled_sum = tf.reduce_sum(x)
        efflen_ladj = tf.reduce_sum(tf.log(self.efflens), axis=1) - n * tf.log(tf.squeeze(x_scaled_sum))
        # efflen_ladj = tf.Print(efflen_ladj, [efflen_ladj], "EFFLEN LADJ", summarize=10)
        hsb_ladj_tensors = []

        # TODO: It may make more sense to build this on the julia side so we
        # can save memory by passing it as a placeholder. Let's just build it
        # here first so we can see if memory use is improved at all.
        # x -> y transformation
        for sample_num in range(num_samples):
            print(sample_num)

            left_child = np.repeat(-1, num_nodes)
            right_child = np.repeat(-1, num_nodes)
            for i in range(1, num_nodes):
                parent_idx = self.node_parent_idxs[i, sample_num] - 1
                if right_child[parent_idx] == -1:
                    right_child[parent_idx] = i
                else:
                    left_child[parent_idx] = i

            # set child indexes
            x_index = np.zeros((n,1), dtype=int)
            for i in range(num_nodes):
                if self.node_js[i, sample_num] != 0:
                    x_index[self.node_js[i, sample_num] - 1] = i

            As = self.As[sample_num]
            x_ = tf.expand_dims(tf.scatter_nd(x_index, x_efflen[sample_num,:], [num_nodes]), -1)
            # x_ = tf.Print(x_, [x_], "X_", summarize=5)
            Axs = [tf.sparse_tensor_dense_matmul(As[0], x_)]
            for i in range(1, len(As)):
                A = As[i]
                Ax_i = tf.sparse_tensor_dense_matmul(A, tf.add(Axs[i-1], x_))
                Axs.append(Ax_i)

            input_values = tf.squeeze(tf.add(tf.reduce_sum(tf.stack(Axs), axis=0), x_))
            # input_values = tf.Print(input_values, [input_values], "INPUT VALUES", summarize=5)

            k = 0
            internal_node_indexes = []
            internal_node_left_indexes = []
            for i in range(num_nodes):
                if self.node_js[i, sample_num] == 0:
                    internal_node_indexes.append(i)
                    internal_node_left_indexes.append(left_child[i])
                    k += 1

            internal_node_values = tf.gather(input_values, internal_node_indexes)
            # internal_node_values = tf.Print(internal_node_values, [input_values], "INTERNAL NODE VALUES")
            hsb_ladj_tensors.append(-tf.reduce_sum(tf.log(internal_node_values)))

            left_node_values = tf.gather(input_values, internal_node_left_indexes)
            # left_node_values = tf.Print(left_node_values, [left_node_values], "LEFT NODE VALUES")
            y_h = tf.divide(tf.to_double(left_node_values),
                            tf.to_double(internal_node_values))
            y_tensors.append(y_h)

            assert(k == n - 1)

        y = tf.stack(y_tensors, name="y")
        y = tf.clip_by_value(y, 1e-10, 1 - 1e-10)

        y_div_omy = tf.divide(y, (1-y))
        y_logit = tf.to_float(tf.log(y_div_omy))
        # y_logit = tf.Print(y_logit, [y_logit], "Y_LOGIT", summarize=10)
        mu = tf.identity(musigma[...,0,:], name="mu")
        # mu = tf.Print(mu, [mu], "MU", summarize=10)
        sigma = tf.identity(musigma[...,1,:], name="sigma")
        # sigma = tf.Print(sigma, [sigma], "SIGMA")

        logit_ladj = tf.to_float(-tf.log(y) - tf.log(1 - y))
        lp_y = logit_ladj - tf.log(sigma * np.sqrt(2*np.pi)) - \
            tf.divide(tf.square(tf.subtract(y_logit, mu)), 2*tf.square(sigma))
        lp = tf.reduce_sum(lp_y, axis=1)
        # lp = tf.Print(lp, [lp], "LP 1", summarize=10)
        lp += efflen_ladj
        # lp = tf.Print(lp, [lp], "LP 2", summarize=10)
        lp += tf.stack(hsb_ladj_tensors)
        # lp = tf.Print(lp, [lp], "LP 3", summarize=10)

        # TODO: not clear if this helps or hurts
        # lp += x_softmax_ladj

        return lp


class RNASeqApproxLikelihood(edward.RandomVariable, RNASeqApproxLikelihoodDist):
    def __init__(self, *args, **kwargs):
        super(RNASeqApproxLikelihood, self).__init__(*args, **kwargs)

