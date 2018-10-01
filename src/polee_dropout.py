
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import edward2 as ed
from polee_approx_likelihood import *
from polee_training import *

def estimate_transcript_expression_dropout(init_feed_dict, num_samples, n, vars, x0_log):

    log_prior = 0.0

    x_dropout_loc = tf.Variable([-15.0], name="x_dropout_prior_loc")
    x_dropout_scale = tf.nn.softplus(tf.Variable([2.0], name="x_dropout_prior_scale"))
    x_dropout_loc = tf.Print(x_dropout_loc, [x_dropout_loc], "x_dropout_loc")
    x_dropout_scale = tf.Print(x_dropout_scale, [x_dropout_scale], "x_dropout_scale")

    # x_dropout_dist = tfd.MultivariateNormalDiag(
    #     loc=tf.ones([1,n]) * x_dropout_loc,
    #     scale_diag=tf.ones([1,n]) * x_dropout_scale)

    x_dropout_dist = tfd.Normal(
        loc=x_dropout_loc,
        scale=x_dropout_scale)

    # x_dropout_prob = tf.Variable(tf.zeros([1,2]), name="x_dropout_prob", trainable=False)
    x_dropout_prob = tf.sigmoid(tf.Variable(0.0, name="x_dropout_prob"))
    # x_dropout_prob = tf.Variable([[-4.0, 4.0]], name="x_dropout_prob", trainable=False)
    # x_dropout_prob = tf.Print(x_dropout_prob, [x_dropout_prob], "x_dropout_prob")

    x_dropout_prob = tf.Print(x_dropout_prob, [x_dropout_prob], "x_dropout_prob")

    # The question here is whether the non-dropout prior
    # should be pooled across transcripts or what.


    x_non_dropout_scale = tf.Print(x_non_dropout_scale,
        [tf.reduce_min(x_non_dropout_scale), tf.reduce_max(x_non_dropout_scale)],
        "x_non_dropout_scale span")

    # TODO: No! We can't set the prior based on x0_log values!
    x_non_dropout_loc_prior = tfd.Normal(
        loc=tf.constant(-8.0, dtype=tf.float32),
        scale=2.0)

    x_non_dropout_loc = tf.Variable(
        # tf.expand_dims(np.mean(x0_log, 0), 0),
        tf.fill([1,n], np.float32(np.quantile(x0_log, 0.95))),
        dtype=tf.float32,
        name="x_non_dropout_loc")

    log_prior += tf.reduce_sum(x_non_dropout_loc_prior.log_prob(x_non_dropout_loc))

    x_non_dropout_loc = tf.Print(x_non_dropout_loc,
        [tf.reduce_min(x_non_dropout_loc), tf.reduce_max(x_non_dropout_loc)],
        "x_non_dropout_loc span")

    # x_non_dropout_dist = tfd.MultivariateNormalDiag(
    #     loc=x_non_dropout_loc,
    #     scale_diag=x_non_dropout_scale)

    x_non_dropout_dist = tfd.Normal(
        loc=x_non_dropout_loc,
        scale=x_non_dropout_scale)

    # print(x_dropout_dist.batch_shape)
    # print(x_non_dropout_dist.batch_shape)

    # print(x_dropout_dist.event_shape)
    # print(x_non_dropout_dist.event_shape)

    # print(tfd.Categorical(logits=x_dropout_prob).event_shape)
    # print(tfd.Categorical(logits=x_dropout_prob).batch_shape)

    # x_prior = tfd.Mixture(
    #     cat=tfd.Categorical(logits=x_dropout_prob),
    #     components=[
    #         x_dropout_dist,
    #         x_non_dropout_dist])

    # x_prior = x_non_dropout_dist

    x = tf.Variable(
        x0_log,
        dtype=tf.float32,
        name="x")

    # x = tf.Print(x, [tf.reduce_min(x), tf.reduce_max(x)], "x span")

    # TODO: I'm afraid this may not work as a mixture due to numerical issues

    dropout_log_prob = tf.log(x_dropout_prob)
    non_dropout_log_prob = tf.log(1.0 - x_dropout_prob)

    x_dropout_log_prob = x_dropout_dist.log_prob(x) + dropout_log_prob
    x_non_dropout_log_prob = x_non_dropout_dist.log_prob(x) + non_dropout_log_prob

    x_dropout_log_prob = tf.Print(x_dropout_log_prob, [x_dropout_log_prob], "x_dropout_log_prob")
    x_non_dropout_log_prob = tf.Print(x_non_dropout_log_prob, [x_non_dropout_log_prob], "x_non_dropout_log_prob")

    x_log_prob = tf.reduce_logsumexp(tf.stack([x_dropout_log_prob, x_non_dropout_log_prob]), 0)
    x_log_prob = tf.Print(x_log_prob, [x_log_prob], "x_log_prob")

    log_prior += tf.reduce_sum(x_log_prob)

    # log_prior += tf.reduce_sum(x_non_dropout_log_prob)

    # x_non_dropout_log_prob = 

    # log_prior += tf.reduce_sum(
    #     x_prior.log_prob(x))

    # log_prior += tf.reduce_sum(
    #     x_non_dropout_dist.log_prob(x))

    # # manual calculation
    # distribution_log_probs = [d.log_prob(x) for d in x_prior.components]
    # distribution_log_probs[0] = tf.Print(distribution_log_probs[0], [distribution_log_probs[0]], "comp log prob 0")
    # distribution_log_probs[1] = tf.Print(distribution_log_probs[1], [distribution_log_probs[1]], "comp log prob 1")

    # cat_log_probs = tf.unstack(tf.nn.softmax(x_dropout_prob), axis=1)
    # final_log_probs = tf.stack([dlp + clp for (dlp, clp) in zip(distribution_log_probs, cat_log_probs)])
    # # final_log_probs = tf.Print(final_log_probs, [final_log_probs], "final_log_probs")
    # x_prior_prob = tf.reduce_logsumexp(final_log_probs, 0)
    # x_prior_prob = tf.Print(x_prior_prob, [x_prior_prob], "x_prior_prob")
    # log_prior += tf.reduce_sum(x_prior_prob)


    log_likelihood = rnaseq_approx_likelihood_from_vars(vars, x)

    log_posterior = log_likelihood + log_prior

    sess = tf.Session()
    train(sess, -log_posterior, init_feed_dict, 100, 5e-2)
