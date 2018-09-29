
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import sys
from tensorflow_probability import edward2 as ed
from polee_approx_likelihood import *
from polee_training import *

SIGMA_ALPHA0 = 0.001
SIGMA_BETA0  = 0.001


def estimate_transcript_mixture(
        init_feed_dict, num_samples, n, vars, x0_log,
        num_mix_components, num_pca_components):

    log_prior = 0.0

    # x_bias
    x_bias_loc0 = np.log(1/n)
    x_bias_scale0 = 4.0
    x_bias_prior = tfd.Normal(
        loc=tf.constant(x_bias_loc0, dtype=tf.float32),
        scale=tf.constant(x_bias_scale0, dtype=tf.float32),
        name="x_bias")

    x_bias = tf.Variable(
        np.mean(x0_log, 0),
        dtype=tf.float32,
        name="x_bias")

    log_prior += tf.reduce_sum(x_bias_prior.log_prob(x_bias))

    # w
    w_prior = tfd.Normal(
        loc=tf.constant(0.0, dtype=tf.float32),
        scale=tf.constant(1.0, dtype=tf.float32),
        name="w_prior")

    w = tf.Variable(
        # tf.random_normal([n, num_pca_components], stddev=0.1),
        tf.zeros([n, num_pca_components]),
        dtype=tf.float32,
        name="w")

    log_prior += tf.reduce_sum(w_prior.log_prob(w))

    # z_mix
    z_mix_prior = tfd.Dirichlet(
        concentration=tf.constant(5.0, shape=[num_mix_components]),
        name="z_mix_prior")

    z_mix = tf.Variable(
        tf.zeros([num_mix_components]),
        dtype=tf.float32,
        name="z_mix")

    # z_mix = tf.Print(z_mix, [tf.nn.softmax(z_mix)], "z_mix", summarize=16)


    # z_mix_log_prob = z_mix_prior.log_prob(tf.nn.softmax(z_mix))
    # z_mix_log_prob = tf.Print(z_mix_log_prob, [z_mix_log_prob], "z_mix_log_prob")
    # log_prior += tf.reduce_sum(z_mix_log_prob)

    log_prior += tf.reduce_sum(z_mix_prior.log_prob(tf.nn.softmax(z_mix)))

    # z_comp_loc
    z_comp_loc_prior = tfd.Normal(
        loc=tf.constant(0.0, dtype=tf.float32),
        scale=tf.constant(1.0, dtype=tf.float32),
        name="z_comp_loc_prior")

    z_comp_loc = tf.Variable(
        # tf.zeros([num_mix_components, num_pca_components]),
        tf.random_normal([num_mix_components, num_pca_components], stddev=0.1),
        name="z_comp_loc")

    # z_comp_loc = tf.Print(z_comp_loc, [z_comp_loc], "z_comp_loc")

    log_prior += tf.reduce_sum(z_comp_loc_prior.log_prob(z_comp_loc))

    # z_comp_scale
    # z_comp_scale_prior = tfd.InverseGamma(
    #     concentration=0.1,
    #     rate=0.1,
    #     name="z_comp_scale_prior")

    z_comp_scale_prior = HalfCauchy(
        loc=0.0,
        scale=0.01,
        name="z_comp_scale_prior")

    z_comp_scale = tf.nn.softplus(tf.Variable(
        # tf.fill([num_mix_components, num_pca_components], -3.0),
        tf.fill([num_mix_components, num_pca_components], 1.0),
        trainable=True,
        name="z_comp_scale"))

    # z_comp_scale = tf.Print(z_comp_scale, [z_comp_scale], "z_comp_scale span",
    #     summarize=4*18)

    log_prior += tf.reduce_sum(z_comp_scale_prior.log_prob(z_comp_scale))

    # z
    z_mix_dist = tfd.Categorical(
        logits=z_mix,
        name="z_mix_dist")

    z_comp_dist = tfd.MultivariateNormalDiag(
        loc=z_comp_loc,
        scale_diag=z_comp_scale,
        name="z_comp_dist")

    z_prior = tfd.MixtureSameFamily(
        mixture_distribution=z_mix_dist,
        components_distribution=z_comp_dist,
        name="z_prior")

    z = tf.Variable(
-       tf.random_normal([num_samples, num_pca_components], stddev=0.1),
        # tf.zeros([num_samples, num_pca_components]),
        dtype=tf.float32,
        name="z")

    log_prior += tf.reduce_sum(z_prior.log_prob(z))

    # sample_scale

    # sample_scale_prior = tfd.Normal(
    #     loc=tf.constant(0.0, dtype=tf.float32),
    #     scale=tf.constant(1.0, dtype=tf.float32))

    # sample_scale = tf.Variable(
    #     tf.zeros([num_samples, 1]),
    #     name="sample_scale")

    # log_prior += tf.reduce_sum(sample_scale_prior.log_prob(sample_scale))

    # x_err_scale
    # x_err_scale_prior = tfd.InverseGamma(
    #     concentration=0.01,
    #     rate=0.01,
    #     name="x_err_scale_prior")

    x_err_scale_prior = HalfCauchy(
        loc=0.0,
        scale=0.1,
        name="x_err_scale_prior")

    x_err_scale = tf.nn.softplus(tf.Variable(
        tf.fill([1,n], -1.0),
        name="x_err_scale"))

    log_prior += tf.reduce_sum(x_err_scale_prior.log_prob(x_err_scale))

    # x_err
    x_err_prior = tfd.MultivariateNormalDiag(
        loc=tf.zeros([1,n]),
        scale_diag=tf.fill([1,n], 8.0))
        # scale_diag=x_err_scale)

    # x_err_prior = tfd.StudentT(
    #     df=1.0,
    #     loc=tf.zeros([1,n]),
    #     scale=x_err_scale)

    x_err = tf.Variable(
        tf.zeros([num_samples, n]),
        name="x_err")

    # x_err = tf.Print(x_err, [tf.reduce_min(x_err), tf.reduce_max(x_err)], "x_err span")

    log_prior += tf.reduce_sum(x_err_prior.log_prob(x_err))

    # x
    x_pca = tf.matmul(z, w, transpose_b=True, name="x_pca")

    # TODO: Add some additive error here.
    # x = tf.add(tf.add(x_pca, x_bias), sample_scale, name="x")
    x = tf.add(tf.add(x_pca, x_bias), x_err, name="x")
    # x = tf.add(x_pca, x_bias, name="x")

    # TODO: what if x were mixed with some low-expression dropout component?

    log_likelihood = rnaseq_approx_likelihood_from_vars(vars, x)

    log_posterior = log_likelihood + log_prior

    sess = tf.Session()
    train(sess, -log_posterior, init_feed_dict, 500, 5e-2)

    # print("z")
    # print(sess.run(z))

    print("z_mix")
    print(sess.run(tf.nn.softmax(z_mix)))

    # print("z_comp_loc")
    # print(sess.run(z_comp_loc))

    # print("z_comp_scale")
    # print(sess.run(z_comp_scale))

    print("comp probs")

    # compute component assignment probabilities
    z_ = z_prior._pad_sample_dims(z)
    log_prob_x = z_comp_dist.log_prob(z_)
    log_mix_prob = tf.nn.log_softmax(
          z_mix_dist.logits, axis=-1)
    log_prob_x += log_mix_prob

    component_probs = sess.run(tf.exp(
        log_prob_x - tf.reduce_logsumexp(log_prob_x, keepdims=True, axis=1)))

    return component_probs, sess.run(w), sess.run(x)

    # TODO: maybe pool x_err variance within genes?
    #    (We're already doing that!!!)
    # TODO: 