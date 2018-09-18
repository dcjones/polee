
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sys
from tensorflow_probability import edward2 as ed
from polee_approx_likelihood import *
from polee_training import *

SIGMA_ALPHA0 = 0.001
SIGMA_BETA0  = 0.001


def estimate_transcript_mixture(init_feed_dict, num_samples, n, vars, x0_log, num_components):

    x0_log_min = np.min(x0_log, 0)
    x0_log_max = np.max(x0_log, 0)
    x0_log_span = x0_log_max - x0_log_min
    # qloc0 = x0_log_min + np.random.uniform(size=(1, num_components, 1)) * x0_log_span
    # qloc0 = x0_log_min = np.expand_dims(np.expand_dims(np.linspace(0, 1, num_components), 1), 0) * x0_log_span

    # qloc0 = np.expand_dims(np.stack([
    #     np.random.uniform(low=0.9, high=1.1, size=(n,)) * \
    #     x0_log[np.random.randint(num_samples)] for _ in range(num_components)]), 0)

    # choice_replace = num_samples > num_components
    # qloc0 = np.expand_dims(x0_log[np.random.choice(range(num_samples), num_components, replace=choice_replace),:], 0)

    qloc0 = np.random.uniform(low=-0.5, high=0.5, size=(1, num_components, n)) + \
            np.mean(x0_log, 0)

    # qloc0 = np.expand_dims(np.stack([
    #     x0_log[4*i,:] for i in range(num_components)]), 0)
        # x0_log[np.random.randint(num_samples)] for _ in range(num_components)]), 0)
    print(qloc0.shape)

    # print([np.random.randint(num_samples) for _ in range(num_components)])

    qloc = tf.Variable(
        qloc0,
        name="qloc",
        dtype=tf.float32)

    qscale = tf.ones((1, 1, n)) * tf.nn.softplus(tf.Variable(
        # tf.fill((1, 1, n), 1.0),
        tf.fill((1,), 10.0),
        # trainable=False,
        name="qscale_softplus"))
    # qscale = tf.Print(qscale, [tf.reduce_min(qscale), tf.reduce_max(qscale)], "qscale span")

    qmix_probs = tf.nn.softmax(
        tf.Variable(
            tf.zeros(num_components),
            trainable=False,
            name="qmix_probs_logit"))
    # qmix_probs = tf.Print(qmix_probs, [qmix_probs], "qmix_probs", summarize=16)

    qx = tf.Variable(x0_log, trainable=False, name="qx")

    log_likelihood = rnaseq_approx_likelihood_from_vars(vars, qx)

    # prior
    mix_probs_prior = tfp.distributions.Dirichlet(
        concentration=5.0 * tf.ones(num_components),
        name="mix_probs")

    # x_mu_mu0    = tf.constant(np.log(1.0/n), shape=[n], dtype=tf.float32)
    # x_mu_sigma0 = tf.constant(4.0, shape=[n], dtype=tf.float32)
    x_mu_mu0    = tf.constant(np.log(1.0/n), dtype=tf.float32)
    x_mu_sigma0 = tf.constant(4.0, dtype=tf.float32)
    loc_prior = tfp.distributions.Normal(
        loc=x_mu_mu0, scale=x_mu_sigma0, name="loc_prior")

    # scale_sq_prior = tfp.distributions.InverseGamma(
    #     concentration=tf.constant(SIGMA_ALPHA0, dtype=tf.float32),
    #     rate=tf.constant(SIGMA_BETA0, dtype=tf.float32),
    #     name="scale_sq_prior")
    scale_sq_prior = HalfCauchy(
        loc=tf.constant(0.10, dtype=tf.float32),
        scale=tf.constant(0.05, dtype=tf.float32),
        name="scale_sq_prior")

    mixture_distribution=tfp.distributions.Categorical(probs=qmix_probs)
    components_distribution=tfp.distributions.MultivariateNormalDiag(loc=qloc, scale_diag=qscale)
    mix_dist = tfp.distributions.MixtureSameFamily(
        mixture_distribution=mixture_distribution,
        components_distribution=components_distribution,
        name="mix")

    # print(mix_dist.batch_shape)
    # print(mix_dist.event_shape)
    # sys.exit()

    # evaluating parts of MixtureSameFamily.log_prob to see where the gradient
    # gets stuck.
    qx_ = mix_dist._pad_sample_dims(qx)
    log_prob_x = mix_dist.components_distribution.log_prob(qx_)
    log_mix_prob = tf.nn.log_softmax(
        mix_dist.mixture_distribution.logits, axis=-1)
    log_mix_dist_prob = tf.reduce_logsumexp(log_prob_x + log_mix_prob, axis=-1)

    tf.summary.histogram("qloc", qloc)
    tf.summary.histogram("qscale", qscale)
    tf.summary.histogram("qmix_probs", qmix_probs)
    tf.summary.histogram("qx", qx)

    # TODO: We need qscale to be able to vary, but right now it can go towards
    # 0 and squish qx values together and get absurd probabilities.
    # Maybe use a different prior? Fixing qscale doesn't seem great.
    # Something like: https://github.com/tensorflow/probability/issues/100

    log_prior = \
        tf.reduce_sum(mix_probs_prior.log_prob(qmix_probs)) + \
        tf.reduce_sum(loc_prior.log_prob(tf.transpose(qloc))) + \
        tf.reduce_sum(scale_sq_prior.log_prob(tf.square(qscale))) + \
        tf.Print(tf.reduce_sum(mix_dist.log_prob(qx)),
            [log_mix_dist_prob], "MIX PROBS", summarize=64)

    tf.summary.scalar(
        "max mix prob", tf.reduce_max(mix_probs_prior.log_prob(qmix_probs)))
    tf.summary.scalar(
        "max loc prob", tf.reduce_max(loc_prior.log_prob(tf.transpose(qloc))))
    tf.summary.scalar(
        "max scale_sq", tf.reduce_max(scale_sq_prior.log_prob(tf.square(qscale))))
    # tf.summary.scalar(
    #     "max x prob", tf.reduce_max(mix_dist.log_prob(qx)))
    tf.summary.scalar(
        "likelihood", log_likelihood)
    tf.summary.histogram("x prob", mix_dist.log_prob(qx))

    # log_posterior = log_prior + log_likelihood
    log_posterior = log_prior

    sess = tf.Session()


    train(sess, -log_posterior, init_feed_dict, 500, 2e-3)

    print(sess.run(log_prob_x))
    print(sess.run(log_mix_prob))
    print(sess.run(log_mix_dist_prob))

    # assignment posterior probabilities
    comp_log_probs_list = []
    qscale_i = tf.squeeze(qscale)
    for (qmix_prob_i, qloc_i) in zip(
            tf.unstack(qmix_probs), tf.unstack(tf.squeeze(qloc))):
        component_dist = tfp.distributions.MultivariateNormalDiag(
            loc=qloc_i, scale_diag=qscale_i)

        comp_log_probs_i = component_dist.log_prob(qx) + tf.log(qmix_prob_i)
        comp_log_probs_list.append(comp_log_probs_i)

    comp_log_probs = tf.stack(comp_log_probs_list)
    comp_probs = sess.run(tf.exp(comp_log_probs - tf.reduce_logsumexp(comp_log_probs, axis=0)))

    print(sess.run(comp_log_probs))
    print(comp_probs)
    print(sess.run(qmix_probs))

    qloc_diff = qloc - qloc0
    print(sess.run(tf.reduce_min(qloc_diff, axis=2)))
    print(sess.run(tf.reduce_max(qloc_diff, axis=2)))

    return comp_probs

    # )
    # print(comp_log_probs)
    # print(sess.run(comp_log_probs))

    # TODO:
    #   Now we want to assign samples to components.
