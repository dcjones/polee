
"""
An implementation of parametric t-SNE.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import sys
import os
from tensorflow_probability import edward2 as ed
from polee_approx_likelihood import *
from polee_training import *

def pairwise_vlr(y):
    delta = tf.expand_dims(y, 0) - tf.expand_dims(y, 1)
    delta_mean, delta_var = tf.nn.moments(delta, axes=[2])

    n = tf.cast(tf.shape(delta)[2], tf.float32)

    # Necessary to deal with numerical instability of sqrt
    eps = 0.01

    delta_norm = tf.sqrt(n * delta_var + eps)
    delta_norm = tf.Print(delta_norm, [delta_norm], "delta_norm", summarize=10)

    return delta_norm

def pairwise_l2(y):
    # assume dimension 0 is the batch shape and dimension 1 the data shape

    delta = tf.expand_dims(y, 0) - tf.expand_dims(y, 1)

    # return tf.norm(delta, axis=2), delta

    # Necessary to deal with numerical instability of sqrt
    eps = 0.01

    delta_square_sum = tf.reduce_sum(tf.square(delta), axis=2)
    delta_norm = tf.sqrt(delta_square_sum + eps)

    return delta_norm



def find_sigmas(x0_log, target_perplexity):
    num_samples = x0_log.shape[0]
    n = x0_log.shape[1]
    sigmas = np.zeros(num_samples)
    # TODO: This is wrong. Denomenator is supposed to be over all non-equal pairs.
    for i in range(num_samples):
        # print(i)
        y = x0_log[i, :]
        delta = np.sqrt(np.sum(np.square(x0_log[i,:] - x0_log), axis=1))
        # print(delta)

        sigma_lower = 1e-2
        sigma_upper = 10.0
        for _ in range(10):
            sigma = (sigma_lower + sigma_upper) / 2
            delta_sig = np.exp(-delta / (2*sigma**2))
            delta_sig[i] = 0.0
            delta_sig_sum = np.sum(delta_sig)

            if delta_sig_sum == 0.0:
                sigma_lower = sigma
                continue

            H = 0.0
            for j in range(num_samples):
                if j == i:
                    continue
                pji = delta_sig[j] / delta_sig_sum
                if pji > 1e-16:
                    H -= pji * np.log2(pji)

            perplexity = 2**H
            # print(perplexity)
            if perplexity > target_perplexity:
                sigma_upper = sigma
            else:
                sigma_lower = sigma
        # print(delta_sig / delta_sig_sum)
        # print((i, perplexity))
        sigmas[i] = (sigma_lower + sigma_upper) / 2
    # print(sigmas)
    return sigmas


def tsne_p(x, sigmas):
    # delta = pairwise_l2(x) # [B, B]
    delta = pairwise_vlr(x) # [B, B]

    # delta = tf.Print(delta, [delta], "delta", summarize=25)

    # delta = tf.Print(delta, [delta], "delta", summarize=25)
    delta_sig = tf.exp(-delta / (2*tf.square(tf.expand_dims(sigmas, 0)))) # [B, B]
    delta_sig = delta_sig - tf.diag(tf.diag_part(delta_sig))
    # delta_sig = tf.Print(delta_sig, [delta_sig], "delta_sig", summarize=25)
    # delta_sig_sum = tf.reduce_sum(delta_sig, axis=0)
    # delta_sig_sum = tf.reduce_sum(delta_sig)

    # Note: pretty sure axis of this sum should be 0, but double check

    # p_j_i = delta_sig / delta_sig_sum
    # p_j_i = tf.Print(p_j_i, [p_j_i], "p_j_i", summarize=25)

    # symmetrize p_j_i
    # B = tf.cast(tf.shape(sigmas), tf.float32)
    # p_ji = (p_j_i + tf.transpose(p_j_i)) / (2*B)

    # TODO: Why not just this
    p_ji = (delta_sig + tf.transpose(delta_sig))
    p_ji /= tf.reduce_sum(p_ji)

    return p_ji


def tsne_q(z, alpha):
    # delta = pairwise_l2(z) # [B, B]
    delta = pairwise_vlr(z) # [B, B]
    delta_sig = tf.pow(1 + delta / alpha, -(alpha+1)/2)

    # delta_sig_sum = tf.reduce_sum(delta_sig, axis=0) - tf.diag_part(delta_sig)
    delta_sig = tf.Print(delta_sig, [tf.reduce_min(delta_sig), tf.reduce_max(delta_sig)], "delta_sig span", summarize=10)
    delta_sig = delta_sig - tf.diag(tf.diag_part(delta_sig))
    delta_sig_sum = tf.reduce_sum(delta_sig)

    q_ji = delta_sig / delta_sig_sum
    q_ji = tf.Print(q_ji, [tf.reduce_min(delta), tf.reduce_max(delta)], "delta span", summarize=10)
    q_ji = tf.Print(q_ji, [delta_sig_sum], "delta_sig_sum")
    return q_ji


def sample_minibatch(feed_dict, full_data, num_samples, B):
    idxs = np.array([i for i in range(num_samples)])
    # np.random.shuffle(idxs) # TODO:
    # idxs = np.flip(idxs)
    minibatch_idx = idxs[0:B]

    for (var, arr) in full_data.items():
        if len(arr.shape) > 1:
            feed_dict[var] = arr[minibatch_idx,:]
        else:
            feed_dict[var] = arr[minibatch_idx]


def estimate_transcript_tsne(
        init_feed_dict, num_samples, n, vars, x0_log,
        num_pca_components, B):

    B = 72
    alpha = 1.0
    target_perplexity = 5.0

    tsne_all_sigmas = find_sigmas(x0_log, target_perplexity)
    tsne_sigmas = tf.placeholder(tf.float32, shape=(None,), name="tsne_sigmas_batch")
    z0_mu = tf.placeholder(tf.float32, (None, n-1))

    full_data = {}
    varnames = ["efflen", "la_mu", "la_sigma", "la_alpha", "left_index", "right_index", "leaf_index"]
    for varname in varnames:
        full_data[vars[varname]] = init_feed_dict[vars[varname]]
    full_data[tsne_sigmas] = tsne_all_sigmas
    full_data[z0_mu] = np.zeros((num_samples, n-1))

    # samples expression values
    x = tf.log(rnaseq_approx_likelihood_sampler(
        z0_mu,
        efflens=vars["efflen"],
        mu=vars["la_mu"],
        sigma=vars["la_sigma"],
        alpha=vars["la_alpha"],
        left_index=vars["left_index"],
        right_index=vars["right_index"],
        leaf_index=vars["leaf_index"]))

    # x = tf.Print(x, [x], "x", summarize=10)

    # feed-forward network mapping onto low-dimensional latent space
    lyr1 = tf.layers.dense(
        x, 256,
        use_bias=False,
        activation=tf.nn.leaky_relu,
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.random_normal_initializer(0.0, 0.0001/n))

    # lyr1 = tf.Print(lyr1, [lyr1], "lyr1", summarize=10)

    # lyr2 = tf.layers.dense(
    #     lyr1, 256,
    #     activation=tf.nn.leaky_relu,
    #     bias_initializer=tf.zeros_initializer(),
    #     kernel_initializer=tf.random_normal_initializer(0.0, 0.1))

    # lyr2 = tf.Print(lyr2, [lyr2], "lyr2", summarize=10)

    z = tf.layers.dense(
        lyr1, num_pca_components,
        use_bias=False,
        activation=tf.identity,
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.random_normal_initializer(0.0, 0.1))

    # weights = tf.get_default_graph().get_tensor_by_name(
    #     os.path.split(z.name)[0] + '/kernel:0')

    # bias = tf.get_default_graph().get_tensor_by_name(
    #     os.path.split(z.name)[0] + '/bias:0')

    # z = tf.Print(z, [weights], "weights", summarize=10)
    # z = tf.Print(z, [bias], "bias", summarize=10)

    # z = tf.Print(z, [z], "z", summarize=10)

    # t-SNE objective function

    # TODO: make sure diagonal on both of these is zeroed out
    one_diag = tf.diag(tf.ones(tf.shape(tsne_sigmas)))
    eps = 0.00001
    p = tsne_p(x, tsne_sigmas) # [B, B]
    p += eps
    # p += one_diag

    # p = tf.Print(p, [p], "p", summarize=10)
    p = tf.Print(p, [tf.reduce_min(p), tf.reduce_max(p)], "p", summarize=10)

    q = tsne_q(z, alpha) # [B, B]
    q += eps
    # q += one_diag

    # q = tf.Print(q, [q], "q", summarize=10)
    q = tf.Print(q, [tf.reduce_min(q), tf.reduce_max(q)], "q", summarize=10)

    loss = tf.reduce_sum(p * tf.log(p / q))
    # loss = tf.Print(loss, [p*tf.log(p/q)], "p*log(p/q)", summarize=400)
    # loss = tf.Print(loss, [tf.reduce_min(tf.abs(p*tf.log(p/q))), tf.reduce_max(tf.abs(p*tf.log(p/q)))], "p*log(p/q)", summarize=400)
    # loss = tf.Print(loss, [tf.reduce_min(tf.abs(tf.log(p/q))), tf.reduce_max(tf.abs(tf.log(p/q)))], "p*log(p/q)", summarize=400)
    # loss = tf.Print(loss, [tf.gradients(loss, q_diff)], "loss grad", summarize=10)

    # TODO: pairwise vlr in data space
    # TODO: pairwise vlr in latent space

    sess = tf.Session()
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    sample_minibatch(init_feed_dict, full_data, num_samples, B)
    sess.run(init, feed_dict=init_feed_dict)

    batch_feed_dict = {}
    for var in full_data:
        batch_feed_dict[var] = None

    sample_minibatch(batch_feed_dict, full_data, num_samples, B)
    for iter in range(500):
        _, loss_value = sess.run([train, loss], feed_dict=batch_feed_dict)
        print(loss_value)

    return sess.run(z, feed_dict=full_data)


if False:
    def estimate_transcript_tsne(
            init_feed_dict, num_samples, n, vars, x0_log,
            num_pca_components):

        log_prior = 0.0

        # x_bias
        # ------
        x_bias_loc0 = np.log(1/n)
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

        # z
        # -
        z = tf.Variable(
            # tf.random_normal([num_samples, num_pca_components], stddev=0.1),
            tf.zeros([num_samples, num_pca_components]),
            name="z")

        # z = tf.Print(z, [z], "z")

        hidden1 = tf.layers.dense(
            z, 256, activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(0.0, 0.01))

        hidden2 = tf.layers.dense(
            hidden1, 64, activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(0.0, 0.01))

        x_loc = x_bias + tf.layers.dense(
            hidden2, n, use_bias=False, activation=tf.identity)

        # x_scale
        x_scale_prior = tfd.InverseGamma(
            0.001, 0.001)

        x_scale_softminus = tf.Variable(
            tf.constant(-3.0, shape=[n], dtype=tf.float32),
            trainable=False,
            name="x_scale")
        x_scale = tf.nn.softplus(x_scale_softminus)
        # x_scale = tf.Print(x_scale,
        #     [tf.reduce_min(x_scale), tf.reduce_max(x_scale)], "x_scale span")

        log_prior += tf.reduce_sum(x_scale_prior.log_prob(x_scale))

        # x
        x_prior = tfd.MultivariateNormalDiag(
            loc=x_loc,
            scale_diag=x_scale,
            name="x_prior")

        print(x_loc)

        x = tf.Variable(
            x0_log,
            dtype=tf.float32,
            trainable=False,
            name="x")

        print(x)

        log_prior += tf.reduce_sum(x_prior.log_prob(x))

        # likelihood
        log_likelihood = rnaseq_approx_likelihood_from_vars(vars, x)
        log_posterior = log_likelihood + log_prior

        # print(tf.trainable_variables())
        # sys.exit()

        # TODO: t-sne regularization
        #
        # the question here really is whether it makes sense to
        # compine the t-SNE objective function with the log-likelihood.
        # Let's start with just a probalistic decoder, then go from there.



        # Pre-train
        sess = tf.Session()
        train(sess, -log_prior, init_feed_dict, 10000, 1e-1)

        # Train
        train(
            sess, -log_posterior, init_feed_dict, 1000, 1e-1,
            var_list=tf.trainable_variables() + [x, x_scale_softminus])

        return sess.run(z)

