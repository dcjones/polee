
import tensorflow as tf
from tensorflow.contrib import distributions
from tensorflow.contrib import framework
import edward


class RNASeqApproxLikelihoodDist(distributions.Distribution):
    def __init__(self, mu, sigma, scale_sigma,
                 validate_args=False,
                 allow_nan_stats=False,
                 name="RNASeqApproxLikelihood"):

        with tf.name_scope(name, values=[mu, sigma, scale_sigma]) as ns:
            self._mu = tf.identity(mu, name="mu")
            self._sigma = tf.identity(sigma, name="sigma")
            self._scale_sigma = tf.identity(scale_sigma, name="scale_sigma")
            framework.assert_same_float_dtype([self._mu, self._sigma,
                                               self._scale_sigma])
        parameters = locals()

        # TODO: Should scale_mu really be 0?
        self._scale_dist = distributions.Normal(0.0, self._scale_sigma)
        self._expr_dist = distributions.MultivariateNormalDiag(self._mu, self._sigma)

        super(RNASeqApproxLikelihoodDist, self).__init__(
              dtype=self._mu.dtype,
              validate_args=validate_args,
              allow_nan_stats=allow_nan_stats,
              is_continuous=True,
              is_reparameterized=False,
              parameters=parameters,
              graph_parents=[self._mu, self._sigma, self._scale_sigma])

    def _get_event_shape(self):
        return tf.TensorShape(self._mu.get_shape()[-1] + 1)

    def _get_batch_shape(self):
        return self._mu.get_shape()[:-1]

    def _log_prob(self, y):
        n = self._mu.get_shape()[-1] + 1
        expy = tf.exp(y)
        scale = tf.reduce_sum(expy, axis=-1)

        # expy_trailing_sum[i] = sum_{k=i}^{n} expy[k]
        expy_trailing_sum = tf.cumsum(expy, axis=-1, reverse=True)[...,:-1]
        scaled_expy = tf.divide(expy[...,:-1], expy_trailing_sum)

        # centering = 1 / (n - i)
        centering = tf.divide(1.0, tf.to_float(tf.range(n - 1, 0, -1)))
        x = tf.log(scaled_expy) - tf.log(1 - scaled_expy) - centering

        ll = self._expr_dist.log_pdf(x)
        scale_lp = self._scale_dist.log_pdf(scale)

        return ll + scale_lp


class RNASeqApproxLikelihood(edward.RandomVariable, RNASeqApproxLikelihoodDist):
    def __init__(self, *args, **kwargs):
        super(RNASeqApproxLikelihood, self).__init__(*args, **kwargs)

