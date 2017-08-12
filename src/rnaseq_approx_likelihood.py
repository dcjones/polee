
import numpy as np
import tensorflow as tf
from tensorflow.contrib import distributions
from tensorflow.contrib import framework
import edward


class RNASeqApproxLikelihoodDist(distributions.Distribution):
    def __init__(self, x, efflens, node_parent_idxs, node_js,
                 validate_args=False,
                 allow_nan_stats=False,
                 name="RNASeqApproxLikelihood"):

        with tf.name_scope(name, values=[x]) as ns:
            self.x = tf.identity(x, name="rnaseq/x")
            framework.assert_same_float_dtype([self.x])
        parameters = locals()

        self.efflens = efflens
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
        n = int(self.x.get_shape()[-1])

        num_samples = self.node_parent_idxs.shape[1]
        num_nodes = self.node_parent_idxs.shape[0]

        y_tensors = []

        # TODO: This shit makes me real uncomfortable. This is not a bijection,
        # and there is no jacobian term. Suggests we are doing things wrong. I
        # could do something like exp(x_i) / (1 + sum(x_i)) which should be a
        # bijection with a well defined jacobian.
        x = tf.nn.softmax(self.x)

        # effective length transform
        x_scaled = tf.multiply(x, self.efflens)
        x_scaled_sum = tf.reduce_sum(x_scaled, axis=1, keep_dims=True)
        x_efflen = tf.divide(x_scaled, x_scaled_sum)
        efflen_ladj = tf.reduce_sum(tf.log(self.efflens), axis=1) - n * tf.log(tf.squeeze(x_scaled_sum))

        # x -> y transformation
        # It's not really practical or efficient to try to build and traverse
        # the HSB tree in tensorflow, but we can accomplish the same thing with
        # some redundant computation using sparse matrix multiplication, which
        # we construct here.
        hsb_ladj_tensors = []
        for h in range(num_samples):
            # set child indexes
            left_child = np.repeat(-1, num_nodes)
            right_child = np.repeat(-1, num_nodes)
            for i in range(1, num_nodes):
                parent_idx = self.node_parent_idxs[i, h]-1
                if right_child[parent_idx] == -1:
                    right_child[parent_idx] = i
                else:
                    left_child[parent_idx] = i

            # arrays of arrays of indexes
            J = [None] * (num_nodes)
            for i in range(num_nodes-1, -1, -1):
                if self.node_js[i, h] == 0:
                    J[i] = J[left_child[i]] + J[right_child[i]]
                else:
                    J[i] = [self.node_js[i, h] - 1]

            entry_count = 0
            for i, js in enumerate(J):
                entry_count += len(js)

            indices = np.zeros([entry_count, 2], dtype=int)
            entry_num = 0
            for i, js in enumerate(J):
                for j in js:
                    indices[entry_num, 0] = i
                    indices[entry_num, 1] = j
                    entry_num += 1

            values = np.ones([entry_count], dtype=np.float32)
            A = tf.SparseTensor(indices, values, [num_nodes, n])

            input_values = tf.squeeze(tf.sparse_tensor_dense_matmul(A, tf.expand_dims(x_efflen[h,:], -1)))

            k = 0
            internal_node_indexes = []
            internal_node_left_indexes = []
            for i in range(num_nodes):
                if self.node_js[i, h] == 0:
                    internal_node_indexes.append(i)
                    internal_node_left_indexes.append(left_child[i])
                    k += 1

            internal_node_values = tf.gather(input_values, internal_node_indexes)
            hsb_ladj_tensors.append(-tf.reduce_sum(tf.log(internal_node_values)))

            y_h = tf.divide(tf.to_double(tf.gather(input_values, internal_node_left_indexes)),
                            tf.to_double(internal_node_values))
            y_tensors.append(y_h)

            assert(k == n - 1)

        y = tf.stack(y_tensors, name="y")
        y = tf.clip_by_value(y, 1e-10, 1 - 1e-10)
        # y = tf.Print(y, [tf.reduce_min(y), tf.reduce_max(y)], "Y SPAN")
        # y = tf.Print(y, [y[0, 0:10]], "Y")

        # TODO: evaluate prob
        # This is just logit-normal

        y_div_omy = tf.divide(y, (1-y))
        y_logit = tf.to_float(tf.log(y_div_omy))
        mu = tf.identity(musigma[...,0,:], name="mu")
        sigma = tf.identity(musigma[...,1,:], name="sigma")

        lp_y = y_logit - tf.log(sigma * np.sqrt(2*np.pi)) - \
            tf.divide(tf.square(tf.subtract(y_logit, mu)), 2*tf.square(sigma))
        lp = tf.reduce_sum(lp_y, axis=1)

        lp += efflen_ladj
        lp += tf.stack(hsb_ladj_tensors)

        return lp


class RNASeqApproxLikelihood(edward.RandomVariable, RNASeqApproxLikelihoodDist):
    def __init__(self, *args, **kwargs):
        super(RNASeqApproxLikelihood, self).__init__(*args, **kwargs)

