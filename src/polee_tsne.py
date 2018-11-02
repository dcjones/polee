
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
from polee_transcript_expression import estimate_transcript_expression
from polee_approx_likelihood import *

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
    print(sigmas)
    # sigmas[:] = 2.0
    # sigmas[:] = 3.0
    return sigmas


def tsne_p(x, sigmas):
    delta = pairwise_l2(x) # [B, B]
    # delta = pairwise_vlr(x) # [B, B]

    # delta = tf.Print(delta, [delta], "delta", summarize=10000)
    # delta = tf.Print(delta, [tf.reduce_min(delta), tf.reduce_max(delta)], "delta")
    # delta = tf.Print(delta, [tf.reduce_all(tf.is_finite(delta))], "delta")

    delta_sig = tf.exp(-delta / (2*tf.square(tf.expand_dims(sigmas, 0)))) # [B, B]
    delta_sig = tf.clip_by_value(delta_sig, 1e-12, 1.0)
    delta_sig = delta_sig - tf.diag(tf.diag_part(delta_sig))

    # delta_sig = tf.Print(delta_sig, [tf.reduce_min(delta_sig), tf.reduce_max(delta_sig)], "delta_sig")
    # delta_sig = tf.Print(delta_sig, [tf.reduce_all(tf.is_finite(delta_sig))], "delta_sig")
    # delta_sig = tf.Print(delta_sig, [delta_sig], "delta_sig", summarize=25)

    delta_sig_sum = tf.reduce_sum(delta_sig, axis=0)

    # delta_sig_sum = tf.Print(delta_sig_sum, [tf.reduce_min(delta_sig_sum), tf.reduce_max(delta_sig_sum)], "delta_sig_sum")
    # delta_sig_sum = tf.Print(delta_sig_sum, [tf.reduce_all(tf.is_finite(delta_sig_sum))], "delta_sig_sum")

    p_j_i = delta_sig / delta_sig_sum
    # p_j_i = tf.Print(p_j_i, [p_j_i], "p_j_i", summarize=25)
    # p_j_i = tf.Print(p_j_i, [tf.reduce_all(tf.is_finite(p_j_i))], "p_j_i")

    # symmetrize p_j_i
    B = tf.cast(tf.shape(sigmas), tf.float32)
    p_ji = (p_j_i + tf.transpose(p_j_i)) / (2*B)

    # Simpler way, that is more sensitive to outliers, according to the paper.
    # p_ji = (delta_sig + tf.transpose(delta_sig))
    # p_ji /= tf.reduce_sum(p_ji)

    # p_ji = tf.Print(p_ji, [tf.reduce_all(tf.is_finite(p_ji))], "p_ji")

    return p_ji


def tsne_q(z, alpha):
    delta = pairwise_l2(z) # [B, B]
    # delta = pairwise_vlr(z) # [B, B]
    delta_sig = tf.pow(1 + delta / alpha, -(alpha+1)/2)

    # delta_sig_sum = tf.reduce_sum(delta_sig, axis=0) - tf.diag_part(delta_sig)
    # delta_sig = tf.Print(delta_sig, [tf.reduce_min(delta_sig), tf.reduce_max(delta_sig)], "delta_sig span", summarize=10)
    delta_sig = delta_sig - tf.diag(tf.diag_part(delta_sig))
    delta_sig_sum = tf.reduce_sum(delta_sig)

    q_ji = delta_sig / delta_sig_sum
    # q_ji = tf.Print(q_ji, [tf.reduce_min(delta), tf.reduce_max(delta)], "delta span", summarize=10)
    # q_ji = tf.Print(q_ji, [delta_sig_sum], "delta_sig_sum")
    return q_ji


def sample_minibatch(feed_dict, full_data, num_samples, B):
    idxs = np.array([i for i in range(num_samples)])
    np.random.shuffle(idxs)
    minibatch_idx = idxs[0:B]

    for (var, arr) in full_data.items():
        if len(arr.shape) > 1:
            feed_dict[var] = arr[minibatch_idx,:]
        else:
            feed_dict[var] = arr[minibatch_idx]


def estimate_tsne(
        x_loc_full, x_scale_full,
        init_feed_dict, vars,
        num_pca_components, B, use_neural_network, sess):

    # assume we don't have to reinitialize anything here
    init_feed_dict = dict()

    num_samples, n = np.shape(x_loc_full)

    # TODO: options to pass these parameters in
    alpha = 1.0
    target_perplexity = 50.0

    x0 = np.random.normal(x_loc_full, x_scale_full)
    tsne_all_sigmas = find_sigmas(x0, target_perplexity)

    # tsne_all_sigmas = find_sigmas(x_loc_full, target_perplexity)

    x_loc = tf.placeholder(tf.float32, (None, n), name="x_loc")
    x_scale = tf.placeholder(tf.float32, (None, n), name="x_scale")
    tsne_sigmas = tf.placeholder(tf.float32, shape=(None,), name="tsne_sigmas_batch")

    full_data = {}
    full_data[tsne_sigmas] = tsne_all_sigmas
    full_data[x_loc] = x_loc_full
    full_data[x_scale] = x_scale_full

    # print(n)
    # print(np.shape(x_loc_full))
    # print(np.shape(x_scale_full))
    # print(x_loc)
    # print(x_scale)
    # print(B)
    # sys.exit()

    x = ed.Normal(loc=x_loc, scale=x_scale, name="x")

    # feed-forward network mapping onto low-dimensional latent space
    if use_neural_network:
        activation=tf.nn.leaky_relu

        lyr1 = tf.layers.dense(
            x, 500,
            activation=activation,
            bias_initializer=tf.zeros_initializer(),
            # kernel_initializer=tf.random_normal_initializer(0.0, 0.0001/n))
        )

        lyr2 = tf.layers.dense(
            lyr1, 500,
            activation=activation,
            bias_initializer=tf.zeros_initializer(),
            # kernel_initializer=tf.random_normal_initializer(0.0, 0.01))
        )

        lyr3 = tf.layers.dense(
            lyr1, 2000,
            activation=activation,
            bias_initializer=tf.zeros_initializer(),
            # kernel_initializer=tf.random_normal_initializer(0.0, 0.01))
        )

        z = tf.layers.dense(
            lyr3,
            num_pca_components,
            activation=tf.identity)
    else:
        z = tf.layers.dense(
            x,
            num_pca_components,
            activation=tf.identity)

    # for numerical stability
    eps = 1e-6

    p = tsne_p(x, tsne_sigmas) # [B, B]
    p += eps
    # p = tf.Print(p, [tf.reduce_min(p), tf.reduce_max(p)], "p")
    # p = tf.Print(p, [p], "p", summarize=16)

    q = tsne_q(z, alpha) # [B, B]
    q += eps
    # q = tf.Print(q, [q], "q", summarize=16)
    # q = tf.Print(q, [tf.reduce_min(q), tf.reduce_max(q)], "q")

    # print(B)
    # print(p)
    # print(q)
    # sys.exit()

    # KL divergence between p and q
    loss = tf.reduce_sum(p * tf.log(p / q))
    # loss = tf.Print(loss, [tf.log(p/q)], "log(p/q)", summarize=16)
    # loss = tf.Print(loss, [p*tf.log(p/q)], "p*log(p/q)", summarize=16)
    # loss = tf.Print(loss, [loss], "loss")

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(loss)
    sample_minibatch(init_feed_dict, full_data, num_samples, B)

    init = tf.global_variables_initializer()
    sess.run(init, feed_dict=init_feed_dict)

    batch_feed_dict = {}
    for var in full_data:
        batch_feed_dict[var] = None
    n_iter = 5000
    prog = Progbar(50, n_iter)
    for iter in range(n_iter):
        sample_minibatch(batch_feed_dict, full_data, num_samples, B)
        _, loss_value = sess.run([train, loss], feed_dict=batch_feed_dict)
        prog.update(iter, loss=loss_value)

    # sample z and average
    z_estimate = np.zeros((num_samples, num_pca_components))
    n_est_iterations = 100
    for iter in range(n_est_iterations):
        z_estimate += sess.run(z, feed_dict=full_data) / n_est_iterations

    # return z_estimate, sess.run(p, feed_dict=full_data), sess.run(q, feed_dict=full_data)
    return z_estimate

