from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf
import math

from collections import OrderedDict
from edward.inferences.monte_carlo import MonteCarlo
from edward.models import Normal, RandomVariable, Uniform
from edward.util import copy


class HMC2(MonteCarlo):
  """Hamiltonian Monte Carlo, also known as hybrid Monte Carlo
  (Duane et al., 1987; Neal, 2011).

  Notes
  -----
  In conditional inference, we infer :math:`z` in :math:`p(z, \\beta
  \mid x)` while fixing inference over :math:`\\beta` using another
  distribution :math:`q(\\beta)`.
  ``HMC`` substitutes the model's log marginal density

  .. math::

    \log p(x, z) = \log \mathbb{E}_{q(\\beta)} [ p(x, z, \\beta) ]
                \\approx \log p(x, z, \\beta^*)

  leveraging a single Monte Carlo sample, where :math:`\\beta^* \sim
  q(\\beta)`. This is unbiased (and therefore asymptotically exact as a
  pseudo-marginal method) if :math:`q(\\beta) = p(\\beta \mid x)`.
  """
  def __init__(self, *args, **kwargs):
    """
    Examples
    --------
    >>> z = Normal(mu=0.0, sigma=1.0)
    >>> x = Normal(mu=tf.ones(10) * z, sigma=1.0)
    >>>
    >>> qz = Empirical(tf.Variable(tf.zeros([500])))
    >>> data = {x: np.array([0.0] * 10, dtype=np.float32)}
    >>> inference = ed.HMC({z: qz}, data)
    """
    super(HMC2, self).__init__(*args, **kwargs)

  def initialize(self, step_size=0.25, n_steps=2, *args, **kwargs):
    """
    Parameters
    ----------
    step_size : float, optional
      Step size of numerical integrator.
    n_steps : int, optional
      Number of steps of numerical integrator.
    """
    self.log_step_size = tf.Variable(math.log(step_size), trainable=False)
    self.log_adapted_step_size = tf.Variable(math.log(step_size), trainable=False)
    self.n_steps = n_steps
    self.target_accept_rate = 0.8
    self.step_size_adapt_iterations = 100
    self.step_size_stabization = 10.0
    self.step_size_shrinkage_point = math.log(10*step_size)
    self.step_size_shrinkage_rate = 0.05
    self.step_size_adapt_rate = 0.75
    self.accept_rate_fit = tf.Variable(0.0, trainable=False)

    self.scope_iter = 0  # a convenient counter for log joint calculations
    return super(HMC2, self).initialize(*args, **kwargs)

  def build_update(self):
    """
    Simulate Hamiltonian dynamics using a numerical integrator.
    Correct for the integrator's discretization error using an
    acceptance ratio.

    Notes
    -----
    The updates assume each Empirical random variable is directly
    parameterized by tf.Variables().
    """
    old_sample = {z: tf.gather(qz.params, tf.maximum(self.t - 1, 0))
                  for z, qz in six.iteritems(self.latent_vars)}
    old_sample = OrderedDict(old_sample)

    # Sample momentum.
    old_r_sample = OrderedDict()
    for z, qz in six.iteritems(self.latent_vars):
      event_shape = qz.get_event_shape()
      normal = Normal(mu=tf.zeros(event_shape), sigma=tf.ones(event_shape))
      old_r_sample[z] = normal.sample()

    # Simulate Hamiltonian dynamics.
    step_size_t = tf.exp(tf.where(self.t < self.step_size_adapt_iterations,
                                  self.log_step_size, self.log_adapted_step_size))
    new_sample, new_r_sample = leapfrog(old_sample, old_r_sample,
                                        step_size_t, self._log_joint,
                                        self.n_steps)

    # Calculate acceptance ratio.
    ratio = tf.reduce_sum([0.5 * tf.reduce_sum(tf.square(r))
                           for r in six.itervalues(old_r_sample)])
    ratio -= tf.reduce_sum([0.5 * tf.reduce_sum(tf.square(r))
                            for r in six.itervalues(new_r_sample)])
    ratio += self._log_joint(new_sample)
    ratio -= self._log_joint(old_sample)
    ratio = tf.minimum(0.0, ratio)

    # Accept or reject sample.
    u = Uniform().sample()
    accept = tf.log(u) < ratio
    sample_values = tf.cond(accept, lambda: list(six.itervalues(new_sample)),
                            lambda: list(six.itervalues(old_sample)))
    if not isinstance(sample_values, list):
      # ``tf.cond`` returns tf.Tensor if output is a list of size 1.
      sample_values = [sample_values]

    sample = {z: sample_value for z, sample_value in
              zip(six.iterkeys(new_sample), sample_values)}

    # Update Empirical random variables.
    assign_ops = []
    for z, qz in six.iteritems(self.latent_vars):
      variable = qz.get_variables()[0]
      assign_ops.append(tf.scatter_update(variable, self.t, sample[z]))

    # Adapt step_size
    t = tf.to_float(self.t + 1)
    accept_rate_fit_delta = tf.divide(1.0, t + self.step_size_stabization)
    accept_rate_fit_update = \
        (1.0 - accept_rate_fit_delta) * self.accept_rate_fit + \
        accept_rate_fit_delta * (self.target_accept_rate - tf.exp(ratio))

    # accept_rate_fit_update = tf.Print(accept_rate_fit_update,
            # [accept_rate_fit_update], "Accept Rate Fit")

    log_step_size_update = \
            self.step_size_shrinkage_point - \
            tf.divide(tf.sqrt(t), self.step_size_shrinkage_rate) * \
            accept_rate_fit_update


    # log_step_size_update = tf.Print(log_step_size_update, [tf.exp(log_step_size_update)], message="Step Size")

    step_size_delta = tf.pow(t, -self.step_size_adapt_rate)

    # step_size_delta = tf.Print(step_size_delta, [step_size_delta],
            # message="Step Size Delta")

    log_adapted_step_size_update = \
            step_size_delta * log_step_size_update + \
            (1.0 - step_size_delta) * self.log_adapted_step_size
    # log_adapted_step_size_update = tf.Print(log_adapted_step_size_update, [tf.exp(log_adapted_step_size_update)], message="Adapted Step Size")

    assign_ops.append(tf.assign(self.accept_rate_fit, accept_rate_fit_update))
    assign_ops.append(tf.assign(self.log_step_size, log_step_size_update))
    assign_ops.append(tf.assign(self.log_adapted_step_size,
                                log_adapted_step_size_update))

    # TODO: change n_steps based on step_size (is this possible?)

    # Increment n_accept (if accepted).
    assign_ops.append(self.n_accept.assign_add(tf.where(accept, 1, 0)))
    return tf.group(*assign_ops)

  def _log_joint(self, z_sample):
    """
    Utility function to calculate model's log joint density,
    log p(x, z), for inputs z (and fixed data x).

    Parameters
    ----------
    z_sample : dict
      Latent variable keys to samples.
    """
    if self.model_wrapper is None:
      self.scope_iter += 1
      scope = 'inference_' + str(id(self)) + '/' + str(self.scope_iter)
      # Form dictionary in order to replace conditioning on prior or
      # observed variable with conditioning on a specific value.
      dict_swap = z_sample.copy()
      for x, qx in six.iteritems(self.data):
        if isinstance(x, RandomVariable):
          if isinstance(qx, RandomVariable):
            qx_copy = copy(qx, scope=scope)
            dict_swap[x] = qx_copy.value()
          else:
            dict_swap[x] = qx

      log_joint = 0.0
      for z in six.iterkeys(self.latent_vars):
        z_copy = copy(z, dict_swap, scope=scope)
        log_joint += tf.reduce_sum(z_copy.log_prob(dict_swap[z]))

      for x in six.iterkeys(self.data):
        if isinstance(x, RandomVariable):
          x_copy = copy(x, dict_swap, scope=scope)
          log_joint += tf.reduce_sum(x_copy.log_prob(dict_swap[x]))
    else:
      x = self.data
      log_joint = self.model_wrapper.log_prob(x, z_sample)

    return log_joint


def leapfrog(z_old, r_old, step_size, log_joint, n_steps):
  z_new = z_old.copy()
  r_new = r_old.copy()

  grad_log_joint = tf.gradients(log_joint(z_new), list(six.itervalues(z_new)))
  for _ in range(n_steps):
    for i, key in enumerate(six.iterkeys(z_new)):
      z, r = z_new[key], r_new[key]
      r_new[key] = r + 0.5 * step_size * grad_log_joint[i]
      z_new[key] = z + step_size * r_new[key]

    grad_log_joint = tf.gradients(log_joint(z_new), list(six.itervalues(z_new)))
    for i, key in enumerate(six.iterkeys(z_new)):
      r_new[key] += 0.5 * step_size * grad_log_joint[i]

  return z_new, r_new
