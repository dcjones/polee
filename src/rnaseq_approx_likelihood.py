
import numpy as np
import tensorflow as tf
from tensorflow.contrib import distributions
from tensorflow.contrib import framework
import edward


class RNASeqApproxLikelihoodDist(distributions.Distribution):
    def __init__(self, x, node_parent_idxs, node_js,
                 validate_args=False,
                 allow_nan_stats=False,
                 name="RNASeqApproxLikelihood"):

        with tf.name_scope(name, values=[x]) as ns:
            self.x = tf.identity(x, name="rnaseq/x")
            framework.assert_same_float_dtype([self.x])
        parameters = locals()

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

    def _log_prob(self, as_bs):
        n = int(self.x.get_shape()[-1])

        num_samples = self.node_parent_idxs.shape[1]
        num_nodes = self.node_parent_idxs.shape[0]

        y_tensors = []
        lp_tensors = []

        x = tf.nn.softmax(self.x)
        # x = tf.Print(x, [tf.reduce_min(x), tf.reduce_max(x)], "X SPAN")
        # x = tf.Print(x, [x[0, 0:10]], "X")
        # x = tf.Print(x, [tf.reduce_sum(x), "X SUM"])

        # x -> y transformation
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

            input_values = tf.squeeze(tf.sparse_tensor_dense_matmul(A, tf.expand_dims(x[h,:], -1)))

            k = 0
            internal_node_indexes = []
            internal_node_left_indexes = []
            for i in range(num_nodes):
                if self.node_js[i, h] == 0:
                    internal_node_indexes.append(i)
                    internal_node_left_indexes.append(left_child[i])
                    k += 1

            internal_node_values = tf.gather(input_values, internal_node_indexes)
            lp_tensors.append(-tf.reduce_sum(tf.log(internal_node_values)))

            y_h = tf.divide(tf.to_double(tf.gather(input_values, internal_node_left_indexes)),
                            tf.to_double(internal_node_values))
            y_tensors.append(y_h)

            assert(k == n - 1)

        y = tf.stack(y_tensors, name="y")
        y = tf.clip_by_value(y, 1e-10, 1 - 1e-10)
        # y = tf.Print(y, [tf.reduce_min(y), tf.reduce_max(y)], "Y SPAN")
        # y = tf.Print(y, [y[0, 0:10]], "Y")

        a = tf.identity(as_bs[...,0,:], name="a")
        b = tf.identity(as_bs[...,1,:], name="b")

        ad = tf.to_double(a)
        bd = tf.to_double(b)

        # y -> z transformation
        z = tf.identity(1.0 - tf.pow(1.0 - tf.pow(y, ad), bd), name="z")
        z = tf.clip_by_value(z, 1e-7, 1 - 1e-7)
        z = tf.to_float(z)
        # z = tf.Print(z, [tf.reduce_min(z), tf.reduce_max(z)], "Z SPAN")
        # z = tf.Print(z, [z[0, 0]], "Z0")


        ia = 1.0 / a
        ib = 1.0 / b
        # ib = tf.Print(ib, [tf.reduce_min(ib), tf.reduce_max(ib)], "IB SPAN")

        omz = 1.0 - z
        # omz = tf.Print(omz, [tf.reduce_min(omz), tf.reduce_max(omz)], "OMZ SPAN")

        pow_omz_ib = tf.pow(tf.to_double(omz), tf.to_double(ib))
        # pow_omz_ib = tf.Print(pow_omz_ib, [tf.reduce_min(pow_omz_ib), tf.reduce_max(pow_omz_ib)], "OMZ^IB SPAN")

        logc = tf.to_float(tf.log1p(-pow_omz_ib))

        lp = tf.stack(lp_tensors, name="lp")
        lp -= tf.reduce_sum((ib - 1.0) * tf.log(omz) + (ia - 1.0) * logc - tf.log(tf.multiply(a, b)))

        # lp = tf.Print(lp, [tf.reduce_min(lp), tf.reduce_max(lp)], "LP HSB SPAN")

        return lp


class RNASeqApproxLikelihood(edward.RandomVariable, RNASeqApproxLikelihoodDist):
    def __init__(self, *args, **kwargs):
        super(RNASeqApproxLikelihood, self).__init__(*args, **kwargs)

