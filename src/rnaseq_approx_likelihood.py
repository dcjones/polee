
import numpy as np
import tensorflow as tf
from tensorflow.contrib import distributions
from tensorflow.contrib import framework
import edward
from queue import Queue
import sys


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
        hsb_ladj_tensors = []

        # TODO: It may make more sense to build this on the julia side so we
        # can save memory by passing it as a placeholder. Let's just build it
        # here first so we can see if memory use is improved at all.
        # x -> y transformation
        for sample_num in range(num_samples):
            print(sample_num)
            # breadth first traversal of the tree, separating out nodes by height

            # set child indexes
            left_child = np.repeat(-1, num_nodes)
            right_child = np.repeat(-1, num_nodes)
            x_index = np.zeros((n,1), dtype=int)
            for i in range(1, num_nodes):
                parent_idx = self.node_parent_idxs[i, sample_num]-1
                if right_child[parent_idx] == -1:
                    right_child[parent_idx] = i
                else:
                    left_child[parent_idx] = i

                if self.node_js[i, sample_num] != 0:
                    x_index[self.node_js[i, sample_num] - 1] = i

            q = Queue()
            q.put((0,0))
            As = []
            i_indexes = []
            j_indexes = []
            last_height = 0
            # passthrough = set()
            while not q.empty():
                if q.queue[0][0] != last_height:
                    indexes = np.transpose(np.array([i_indexes, j_indexes]))
                    values = np.ones([len(indexes)], dtype=np.float32)
                    A = tf.SparseTensor(indexes, values, [num_nodes, num_nodes])
                    As.append(A)

                    i_indexes.clear()
                    j_indexes.clear()
                    last_height += 1

                height, i = q.get()
                assert(height == last_height)

                if left_child[i] != -1:
                    j = left_child[i]
                    i_indexes.append(i)
                    j_indexes.append(j)
                    q.put((height+1, j))

                if right_child[i] != -1:
                    j = right_child[i]
                    i_indexes.append(i)
                    j_indexes.append(j)
                    q.put((height+1, j))

            if len(i_indexes) > 0:
                for i in passthrough:
                    i_indexes.append(i)
                    j_indexes.append(i)
                indexes = np.transpose(np.array([i_indexes, j_indexes]))
                values = np.ones([len(indexes)], dtype=np.float32)
                A = tf.SparseTensor(indexes, values, [num_nodes, num_nodes])
                As.append(A)

            As.reverse()
            x_ = tf.expand_dims(tf.scatter_nd(x_index, x_efflen[sample_num,:], [num_nodes]), -1)
            Axs = [tf.sparse_tensor_dense_matmul(As[0], x_)]
            for i in range(1, len(As)):
                A = As[i]
                Ax_i = tf.add(tf.sparse_tensor_dense_matmul(A, Axs[i-1]),
                        tf.sparse_tensor_dense_matmul(A, x_))
                Axs.append(Ax_i)

            input_values = tf.squeeze(tf.reduce_sum(tf.stack(Axs), axis=0))

            k = 0
            internal_node_indexes = []
            internal_node_left_indexes = []
            for i in range(num_nodes):
                if self.node_js[i, sample_num] == 0:
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

