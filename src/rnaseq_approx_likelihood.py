
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
        n = self.x.get_shape()[-1]

        num_samples = self.node_parent_idxs.shape[1]
        num_nodes = self.node_parent_idxs.shape[0]

        print(num_samples)
        print(num_nodes)

        y_tensors = []

        lp = tf.constant(0.0)

        # x -> y transformation
        for h in range(num_samples):
            # set child indexes
            print("setting children...")
            left_child = np.repeat(-1, num_nodes)
            right_child = np.repeat(-1, num_nodes)
            for i in range(1, num_nodes):
                parent_idx = self.node_parent_idxs[i, h]-1
                if right_child[parent_idx] == 0:
                    right_child[parent_idx] = i
                else:
                    left_child[parent_idx] = i
            print("done.")

            input_value = [None] * num_nodes
            y_h = [None] * (n-1)

            print("building tree...")
            k = n - 2
            for i in range(num_nodes-1, -1, -1):
                print(i)
                # leaf node
                if self.node_js[i, h] != 0:
                    input_value[i] = self.x[h, self.node_js[i, h]-1]
                else:
                    input_value[i] = input_value[left_child[i]] + input_value[right_child[i]]
                    y_h[k] = input_value[left_child[i]] / input_value[i]
                    lp -= tf.log(input_value[i])
                    k -= 1
            print("done.")

            assert(k == -1)
            y_tensors.append(tf.stack(y_h))

        y = tf.stack(y_tensors)

        a = as_bs[...,0,:]
        b = as_bs[...,1,:]

        # y -> z transformation
        z = 1.0 - tf.pow(1.0 - tf.pow(y, a), b)

        ia = 1.0 / a
        ib = 1.0 / b
        omz = 1.0 - z
        c = 1.0 - tf.pow(omz, ib)
        lp -= tf.reduce_sum((ib - 1.0) * tf.log(omz) + (ia - 1.0) * tf.log(c) - tf.log(tf.mul(a, b)))
        return lp


class RNASeqApproxLikelihood(edward.RandomVariable, RNASeqApproxLikelihoodDist):
    def __init__(self, *args, **kwargs):
        super(RNASeqApproxLikelihood, self).__init__(*args, **kwargs)

