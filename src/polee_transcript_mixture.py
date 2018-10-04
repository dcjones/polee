
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



def dropout_expression_model(vars, n, x0_log, z_mix, num_mix_components, num_pca_components, x_nondrop_loc):
    log_prior = 0.0

    # x_nondrop_err_scale
    x_nondrop_scale_prior = HalfCauchy(
        loc=0.0,
        scale=0.1,
        name="x_nondrop_scale_prior")

    x_nondrop_scale = tf.nn.softplus(tf.Variable(
        tf.fill([1,n], -1.0),
        name="x_nondrop_scale"))

    log_prior += tf.reduce_sum(x_nondrop_scale_prior.log_prob(x_nondrop_scale))

    # x non-dropout distribution
    x_nondrop_dist = tfd.Normal(
        loc=x_nondrop_loc,
        scale=x_nondrop_scale)

    # x dropout distribution
    x_drop_loc0 = np.float32(np.log(0.001/n))
    x_drop_loc = tf.Variable([x_drop_loc0], name="x_drop_prior_loc")
    x_drop_scale = tf.nn.softplus(tf.Variable([5.0], name="x_drop_prior_scale"))
    x_drop_loc = tf.Print(x_drop_loc, [x_drop_loc], "x_drop_loc")
    x_drop_scale = tf.Print(x_drop_scale, [x_drop_scale], "x_drop_scale")

    x_drop_dist = tfd.Normal(
        loc=x_drop_loc,
        scale=x_drop_scale)

    # x dropout probability
    z_drop_prob = tf.Variable(
        tf.random_normal([num_mix_components, n]),
        name="z_drop_prob")

    # [n]
    x_drop_prob = \
        tf.sigmoid(tf.reduce_sum(z_drop_prob * tf.expand_dims(tf.nn.softmax(z_mix), -1), 0))

    x_drop_prob = tf.Print(x_drop_prob, [
        tf.reduce_min(x_drop_prob), tf.reduce_max(x_drop_prob)], "x_drop_prob span")

    # x_drop_prob = tf.sigmoid(tf.Variable(0.0, name="x_drop_prob"))
    # x_drop_prob = tf.Print(x_drop_prob, [x_drop_prob], "x_drop_prob")

    # mixture log-probability
    x = tf.Variable(
        x0_log,
        dtype=tf.float32,
        name="x")

    drop_log_prob = tf.log(x_drop_prob)
    nondrop_log_prob = tf.log(1.0 - x_drop_prob)

    x_drop_log_prob = x_drop_dist.log_prob(x) + drop_log_prob
    x_nondrop_log_prob = x_nondrop_dist.log_prob(x) + nondrop_log_prob

    # [num_samples, n]
    x_drop_total_log_prob = \
        tf.reduce_logsumexp(tf.stack([x_drop_log_prob, x_nondrop_log_prob]), 0)
    x_drop_posterior_prob = \
        tf.exp(x_drop_log_prob - x_drop_total_log_prob)
    x_nondrop_posterior_prob = \
        tf.exp(x_nondrop_log_prob - x_drop_total_log_prob)

    z_drop_prob = \
        tf.sigmoid(z_drop_prob) * tf.expand_dims(x_drop_posterior_prob, 1) + \
        (1 - tf.sigmoid(z_drop_prob)) * tf.expand_dims(x_nondrop_posterior_prob, 1)

    z_drop_post_log_prob = \
        tf.log(z_drop_prob) + \
        tf.expand_dims(tf.nn.softmax(z_mix), -1)

    z_drop_post_log_prob = tf.reduce_sum(z_drop_post_log_prob, 2)

    z_drop_post_log_prob -= \
        tf.reduce_logsumexp(z_drop_post_log_prob, 1, keepdims=True)


    # x_drop_log_prob = tf.Print(x_drop_log_prob, [x_drop_log_prob], "x_drop_log_prob")
    # x_nondrop_log_prob = tf.Print(x_nondrop_log_prob, [x_nondrop_log_prob], "x_nondrop_log_prob")

    x_drop_log_prob = tf.Print(x_drop_log_prob, [tf.reduce_min(x_drop_log_prob), tf.reduce_max(x_drop_log_prob)], "x_drop_log_prob")
    x_nondrop_log_prob = tf.Print(x_nondrop_log_prob, [tf.reduce_min(x_nondrop_log_prob), tf.reduce_max(x_nondrop_log_prob)], "x_nondrop_log_prob")

    x_log_prob = tf.reduce_logsumexp(tf.stack([x_drop_log_prob, x_nondrop_log_prob]), 0)
    x_log_prob = tf.Print(x_log_prob, [x_log_prob], "x_log_prob")

    log_prior += tf.reduce_sum(x_log_prob)

    return x, log_prior, z_drop_post_log_prob


def estimate_transcript_mixture(
        init_feed_dict, num_samples, n, vars, x0_log,
        num_mix_components, num_pca_components):

    log_prior = 0.0

    # x_bias
    x_bias_loc0 = np.log(1000 * 1/n)
    x_bias_scale0 = 4.0
    x_bias_prior = tfd.Normal(
        loc=tf.constant(x_bias_loc0, dtype=tf.float32),
        scale=tf.constant(x_bias_scale0, dtype=tf.float32),
        name="x_bias")

    x_bias = tf.Variable(
        np.maximum(np.mean(x0_log, 0), x_bias_loc0),
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
        # trainable=False,
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

    z_comp_scale = tf.clip_by_value(tf.nn.softplus(tf.Variable(
        # tf.fill([num_mix_components, num_pca_components], -3.0),
        tf.fill([num_mix_components, num_pca_components], 1.0),
        # trainable=False,
        name="z_comp_scale")), 0.02, 100.0)

    # z_comp_scale = tf.matmul(
    #     tf.nn.softplus(tf.Variable(
    #         # tf.fill([num_mix_components, num_pca_components], -3.0),
    #         tf.fill([num_mix_components, 1], 1.0),
    #         # trainable=False,
    #         name="z_comp_scale")),
    #     tf.ones([1, num_pca_components]))

    z_comp_scale = tf.Print(z_comp_scale, [tf.reduce_min(z_comp_scale), tf.reduce_max(z_comp_scale)], "z_comp_scale span")

    # z_comp_scale = tf.Print(z_comp_scale, [z_comp_scale], "z_comp_scale span",
    #     summarize=4*18)

    log_prior += tf.reduce_sum(z_comp_scale_prior.log_prob(z_comp_scale))

    # z
    z_mix_dist = tfd.Categorical(
        logits=z_mix,
        name="z_mix_dist")

    z_comp_dist = tfd.Normal(
        loc=z_comp_loc,
        scale=z_comp_scale,
        name="z_comp_dist")

    # z_prior = tfd.MixtureSameFamily(
    #     mixture_distribution=z_mix_dist,
    #     components_distribution=z_comp_dist,
    #     name="z_prior")

    z = tf.Variable(
-       tf.random_normal([num_samples, num_pca_components], stddev=0.1),
        # tf.zeros([num_samples, num_pca_components]),
        dtype=tf.float32,
        name="z")

    # num_samples x num_mix_comps x num_pca_components
    z_comp_log_prob = z_comp_dist.log_prob(tf.expand_dims(z, 1))
    z_comp_log_prob += tf.expand_dims(tf.expand_dims(tf.nn.softmax(z_mix), 0), -1)
    print(z_comp_log_prob)

    # num_samples x num_pca_components
    z_log_prob = tf.reduce_logsumexp(z_comp_log_prob, 1)
    print(z_log_prob)

    log_prior += tf.reduce_sum(z_log_prob)

    # log_prior += tf.reduce_sum(z_prior.log_prob(z))

    x_pca = tf.matmul(z, w, transpose_b=True, name="x_pca")

    x_pca = tf.Print(x_pca, [tf.reduce_min(x_pca), tf.reduce_max(x_pca)], "x_pca span")
    x_bias = tf.Print(x_bias, [tf.reduce_min(x_bias), tf.reduce_max(x_bias)], "x_bias span")

    # x (expression)
    x_nondrop_loc = tf.add(x_pca, x_bias, name="x_loc")
    x_nondrop_loc = tf.Print(x_nondrop_loc, [tf.reduce_min(x_nondrop_loc), tf.reduce_max(x_nondrop_loc)], "x_nondrop_loc")

    x, expr_log_prior, z_drop_post_log_prob = dropout_expression_model(
        vars, n, x0_log, z_mix, num_mix_components, num_pca_components, x_nondrop_loc)
    log_prior += expr_log_prior

    log_likelihood = rnaseq_approx_likelihood_from_vars(vars, x)
    log_posterior = log_likelihood + log_prior

    sess = tf.Session()
    train(sess, -log_posterior, init_feed_dict, 50, 5e-2)

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

    # dropout component contribution

    # TODO: for every transcript and sample compute dropout probability
    # TODO: 

    z_drop_post_log_prob = tf.Print(
        z_drop_post_log_prob, [
            tf.reduce_min(z_drop_post_log_prob),
            tf.reduce_max(z_drop_post_log_prob)],
        "z_drop_post_log span")

    # num_samples x num_mix_components
    z_comp_log_prob = tf.reduce_sum(z_comp_log_prob, 2)

    z_comp_log_prob += z_drop_post_log_prob
    component_probs = sess.run(tf.exp(
        z_comp_log_prob - tf.reduce_logsumexp(z_comp_log_prob, 1, keepdims=True)))

    # component_probs = sess.run(tf.exp(
    #     z_drop_post_log_prob +
    #     z_comp_log_prob - tf.reduce_logsumexp(z_comp_log_prob, 1, keepdims=True)))


    # tf.reduce_logsumexp(z_comp_log_prob, 

    # print(z_log_prob - tf.reduce_logsumexp(z_log_prob, 1, keepdims=True))

    # component_probs = sess.run(tf.exp(
    #     z_log_prob - tf.reduce_logsumexp(z_log_prob, 1, keepdims=True)))

    print(component_probs)

    # z_ = z_prior._pad_sample_dims(z)
    # log_prob_x = z_comp_dist.log_prob(z_)
    # log_mix_prob = tf.nn.log_softmax(
    #       z_mix_dist.logits, axis=-1)
    # log_prob_x += log_mix_prob

    # component_probs = sess.run(tf.exp(
    #     log_prob_x - tf.reduce_logsumexp(log_prob_x, keepdims=True, axis=1)))

    return component_probs, sess.run(w), sess.run(x)

    # TODO: maybe pool x_err variance within genes?
    #    (We're already doing that!!!)
    # TODO: 