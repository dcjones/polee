
import tensorflow as tf
from tensorflow.contrib import distributions
from tensorflow.contrib import framework
import edward


class RNASeqApproxLikelihoodDist(distributions.Distribution):
    def __init__(self, y,
                 validate_args=False,
                 allow_nan_stats=False,
                 name="RNASeqApproxLikelihood"):

        with tf.name_scope(name, values=[y]) as ns:
            # y = tf.verify_tensor_all_finite(y, "RNASeqApproxLikelihoodDist: non-finite y")
            self.y = tf.identity(y, name="y")
            framework.assert_same_float_dtype([self.y])
        parameters = locals()

        super(RNASeqApproxLikelihoodDist, self).__init__(
              dtype=self.y.dtype,
              validate_args=validate_args,
              allow_nan_stats=allow_nan_stats,
              is_continuous=True,
              is_reparameterized=False,
              parameters=parameters,
              graph_parents=[self.y,])

    def _get_event_shape(self):
        print("y.get_shape()")
        print(self.y.get_shape())
        return tf.TensorShape([2, self.y.get_shape()[-1] - 1])

    def _get_batch_shape(self):
        print("y.get_shape()")
        print(self.y.get_shape())
        return self.y.get_shape()[:-1]

    def _log_prob(self, musigma):
        n = self.y.get_shape()[-1]
        expy = tf.exp(self.y)
        # expy = tf.verify_tensor_all_finite(expy, "RNASeqApproxLikelihoodDist: non-finite expy")

        # expy_trailing_sum[i] = sum_{k=i}^{n} expy[k]
        expy_trailing_sum = tf.cumsum(expy, axis=-1, reverse=True)[...,:-1]
        # expy_trailing_sum = tf.stack([tf.cumsum(expy_i, axis=-1, reverse=True)
                                      # for expy_i in tf.unstack(expy)])[...,:-1]
        # expy_trailing_sum = tf.check_numerics(expy_trailing_sum,
                # message="RNASeqApproxLikelihoodDist: non-finite expy_trailing_sum", name="B")
        # expy_trailing_sum = tf.verify_tensor_all_finite(expy_trailing_sum, "RNASeqApproxLikelihoodDist: non-finite expy_trailing_sum")

        scaled_expy = tf.divide(expy[...,:-1], expy_trailing_sum)
        # scaled_expy = tf.verify_tensor_all_finite(scaled_expy, "RNASeqApproxLikelihoodDist: non-finite scaled_expy")

        # centering = log(1 / (n - i))
        centering = tf.log(tf.divide(1.0, tf.to_float(tf.range(n - 1, 0, -1))))
        x = tf.log(scaled_expy) - tf.log(1 - scaled_expy) - centering
        # x = tf.verify_tensor_all_finite(x, "RNASeqApproxLikelihoodDist: non-finite x")

        mu    = musigma[...,0,:]
        sigma = musigma[...,1,:]

        # x = tf.check_numerics(x, message="RNASeqApproxLikelihoodDist: non-finite x", name="D")

        # x = tf.Print(x, [tf.reduce_min(x), tf.reduce_max(x)], "x-span")
        ll = distributions.MultivariateNormalDiag(mu, sigma).log_pdf(x)

        # ll = tf.verify_tensor_all_finite(ll, "RNASeqApproxLikelihoodDist: non-finite ll")
        # ll = tf.Print(ll, [tf.reduce_sum(ll)], "Log-likelihood")

        return ll


class RNASeqApproxLikelihood(edward.RandomVariable, RNASeqApproxLikelihoodDist):
    def __init__(self, *args, **kwargs):
        super(RNASeqApproxLikelihood, self).__init__(*args, **kwargs)

