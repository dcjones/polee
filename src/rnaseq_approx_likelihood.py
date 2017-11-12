
import numpy as np
import tensorflow as tf
from tensorflow.contrib import distributions
from tensorflow.contrib import framework
import edward
from queue import Queue
import sys


class RNASeqApproxLikelihoodDist(distributions.Distribution):
    def __init__(self, x, efflens, invhsb_params, node_parent_idxs, node_js,
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
        self.invhsb_params = invhsb_params
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

    def _log_prob(self, laparam):

        n = int(self.x.get_shape()[-1])

        mu    = tf.identity(laparam[...,0,:], name="mu")
        sigma = tf.identity(laparam[...,1,:], name="sigma")
        alpha = tf.identity(laparam[...,2,:], name="alpha")

        num_samples = len(self.invhsb_params)
        num_nodes = self.node_js.shape[0]

        y_tensors = []

        # TODO: This shit makes me real uncomfortable. This is not a bijection,
        # and there is no jacobian term. Suggests we are doing things wrong. I
        # could do something like exp(x_i) / (1 + sum(x_i)) which should be a
        # bijection with a well defined jacobian.
        # self_x = tf.Print(self.x, [tf.reduce_min(self.x), tf.reduce_max(self.x)], "X SPAN")

        # TODO: consider a R^n -> Delta^{n-1} x R transformation, where the extra
        # number is some kind of scale than we can have a prior over.

        # self.x = tf.Print(self.x, [tf.reduce_min(self.x, axis=1)], "X SPAN MIN", summarize=6)
        # self.x = tf.Print(self.x, [tf.reduce_max(self.x, axis=1)], "X SPAN MAX", summarize=6)

        # print("HERE")
        # print(tf.reduce_sum(tf.exp(self.x), axis=1))
        # sys.exit()

        # self.x = tf.Print(self.x,
        #     [tf.reduce_min(tf.reduce_sum(tf.exp(self.x), axis=1)),
        #      tf.reduce_max(tf.reduce_sum(tf.exp(self.x), axis=1))], "X SCALE SPAN")
        # self.x = tf.Print(self.x, [self.x], "X")
        self.x = tf.Print(self.x, [tf.reduce_sum(tf.exp(self.x), axis=1)], "X SCALE", summarize=6)

        x = tf.nn.softmax(self.x)

        # effective length transform
        # --------------------------

        x_scaled = x * self.efflens
        x_scaled_sum = tf.reduce_sum(x_scaled, axis=1, keep_dims=True)
        x_efflen = x_scaled / x_scaled_sum

        efflen_ladj = tf.reduce_sum(tf.log(self.efflens), axis=1) - n * tf.log(tf.squeeze(x_scaled_sum))

        hsb_ladj_tensors = []

        # Inverse hierarchical stick breaking transform
        # ---------------------------------------------

        # TODO: It may make more sense to build this on the julia side so we
        # can save memory by passing it as a placeholder. Let's just build it
        # here first so we can see if memory use is improved at all.
        # x -> y transformation

        # TODO: A conceivably faster way we could do this is with temporary
        # mutable tensors, accumulating values with scatter_add. If we do that,
        # we'd have to manually compute gradients. Pretty tricky, but could be
        # worth pursuing.

        for sample_num in range(num_samples):
            print(sample_num)

            As = self.invhsb_params[sample_num][0]
            x_index = self.invhsb_params[sample_num][1]
            internal_node_indexes = self.invhsb_params[sample_num][2]
            internal_node_left_indexes = self.invhsb_params[sample_num][3]

            input_values = tf.scatter_nd(tf.expand_dims(x_index, -1), x_efflen[sample_num,:], [num_nodes])

            for i in range(len(As)):
                # NOTE: This works under the assumption that scatter_nd adds
                # duplicate entries, which is currently true but not a garuntee of the API.
                # See: https://github.com/tensorflow/tensorflow/issues/8102
                input_values += tf.scatter_nd(tf.expand_dims(As[i][0], -1),
                                              tf.gather_nd(input_values, tf.expand_dims(As[i][1], -1)),
                                              [num_nodes])


            # input_values = tf.Print(input_values, [tf.reduce_min(input_values), tf.reduce_max(input_values)],
            #                         "INPUT VALUES SPAN " + str(i))

            input_values = tf.to_double(input_values)
            input_values = tf.clip_by_value(input_values, 1e-10, 1.0 - 1e-10)

            internal_node_values = tf.gather(input_values, internal_node_indexes)
            hsb_ladj_tensor = tf.log(internal_node_values)
            hsb_ladj_tensors.append(tf.to_float(-tf.reduce_sum(hsb_ladj_tensor)))

            left_node_values = tf.gather(input_values, internal_node_left_indexes)
            y_h = tf.divide(left_node_values, internal_node_values)
            y_tensors.append(y_h)

        hsb_ladj = tf.stack(hsb_ladj_tensors)

        y = tf.stack(y_tensors, name="y")
        y = tf.clip_by_value(y, 1e-10, 1 - 1e-10)

        # logit (inverse logistic) transform
        # ----------------------------------

        y_log = tf.log(y)
        y_om_log = tf.log(1.0 - y)
        y_logit = tf.to_float(y_log - y_om_log)
        y_logit_ladj = tf.reduce_sum(tf.to_float(-y_log - y_om_log), axis=1)


        # normal standardization transform
        # --------------------------------

        z_std = tf.divide(tf.subtract(y_logit, mu), sigma)
        z_std_ladj = -tf.reduce_sum(tf.log(sigma), axis=1)

        # inverse sinh-asinh transform
        # ----------------------------

        z_c = tf.subtract(tf.asinh(z_std), alpha)
        z = tf.sinh(z_c)

        z_ladj = tf.reduce_sum(
            tf.subtract(tf.log(tf.cosh(z_c)),
                        tf.multiply(0.5, tf.log1p(tf.square(z_std)))),
            axis=1)

        # standand normal log-probability
        # -------------------------------

        lp = tf.reduce_sum((-np.log(2.0*np.pi) - tf.square(z)) / 2.0, axis=1)

        # TODO: Really there should be jacobian terms, but this seems to only
        # cause problems. I don't know why.
        # return lp + z_ladj + z_std_ladj + y_logit_ladj + \
        #     hsb_ladj + efflen_ladj

        return lp

class RNASeqApproxLikelihood(edward.RandomVariable, RNASeqApproxLikelihoodDist):
    def __init__(self, *args, **kwargs):
        super(RNASeqApproxLikelihood, self).__init__(*args, **kwargs)

