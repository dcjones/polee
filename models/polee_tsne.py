
"""
An implementation of parametric t-SNE.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import sys
from tensorflow_probability import edward2 as ed
from polee_approx_likelihood import *
from polee_training import *
from polee_transcript_expression import estimate_transcript_expression
from polee_approx_likelihood import *

def pairwise_vlr(y):
    delta = tf.expand_dims(y, 0) - tf.expand_dims(y, 1)
    delta_mean, delta_var = tf.nn.moments(delta, axes=[2])

    n = tf.cast(tf.shape(delta)[2], tf.float32)

    return delta_var

    # Necessary to deal with numerical instability of sqrt
    # eps = 0.01
    # delta_norm = tf.sqrt(n * delta_var + eps)
    # delta_norm = tf.Print(delta_norm, [delta_norm], "delta_norm", summarize=10)

    # return delta_norm

def pairwise_l2_sq(y):
    # assume dimension 0 is the batch shape and dimension 1 the data shape

    delta = tf.expand_dims(y, 0) - tf.expand_dims(y, 1)

    # return tf.norm(delta, axis=2), delta

    # Necessary to deal with numerical instability of sqrt
    eps = 0.01

    delta_square_sum = tf.reduce_sum(tf.square(delta), axis=2)
    # delta_norm = tf.sqrt(delta_square_sum + eps)

    return delta_square_sum



def find_sigmas(x0_log, target_perplexity, use_vlr=False):
    num_samples = x0_log.shape[0]
    n = x0_log.shape[1]
    sigmas = np.zeros(num_samples)
    for i in range(num_samples):
        y = x0_log[i, :]
        if use_vlr:
            delta = np.var(x0_log[i,:] - x0_log, axis=1)
        else:
            delta = np.sqrt(np.sum(np.square(x0_log[i,:] - x0_log), axis=1))

        sigma_lower = 1e-2
        # sigma_upper = 10.0
        sigma_upper = 10.0 * np.sqrt(np.max(delta))
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
            if perplexity > target_perplexity:
                sigma_upper = sigma
            else:
                sigma_lower = sigma
        sigmas[i] = (sigma_lower + sigma_upper) / 2
    # sigmas[:] = 2.0
    # sigmas[:] = 3.0
    return sigmas


# def tsne_p_denomenators(x0, sigmas):
#     delta = pairwise_l2_sq(x0)
#     delta_sig = tf.exp(-delta / (2*tf.square(tf.expand_dims(sigmas, 0)))) # [B, B]
#     delta_sig = tf.clip_by_value(delta_sig, 1e-12, 1.0)
#     delta_sig = delta_sig - tf.diag(tf.diag_part(delta_sig))
#     delta_sig_sum = tf.reduce_sum(delta_sig, axis=0)

#     return delta_sig_sum


def tsne_p(x, sigmas, use_vlr=False):
    if use_vlr:
        delta = pairwise_vlr(x) # [B, B]
    else:
        delta = pairwise_l2_sq(x) # [B, B]

    tf.summary.histogram("delta", delta)
    # delta = tf.Print(delta, [delta], "delta", summarize=25)

    delta_sig = tf.exp(-delta / (2*tf.square(tf.expand_dims(sigmas, 0)))) # [B, B]
    delta_sig = tf.clip_by_value(delta_sig, 1e-12, 1.0)
    delta_sig = delta_sig - tf.diag(tf.diag_part(delta_sig))

    # delta_sig = tf.Print(delta_sig, [delta_sig], "delta_sig", summarize=25)

    delta_sig_sum = tf.reduce_sum(delta_sig, axis=0)

    p_j_i = delta_sig / delta_sig_sum

    # symmetrize p_j_i
    B = tf.cast(tf.shape(sigmas), tf.float32)
    p_ji = (p_j_i + tf.transpose(p_j_i)) / (2*B)

    # Simpler way, that is more sensitive to outliers, according to the paper.
    # p_ji = (delta_sig + tf.transpose(delta_sig))
    # p_ji /= tf.reduce_sum(p_ji)

    # p_ji = tf.Print(p_ji, [tf.reduce_all(tf.is_finite(p_ji))], "p_ji")

    return p_ji


def tsne_q(z, alpha, use_vlr=False):
    if use_vlr:
        delta = pairwise_vlr(z) # [B, B]
    else:
        delta = pairwise_l2_sq(z) # [B, B]
    delta_sig = tf.pow(1 + delta / alpha, -(alpha+1)/2)

    delta_sig = delta_sig - tf.diag(tf.diag_part(delta_sig))
    delta_sig_sum = tf.reduce_sum(delta_sig)

    q_ji = delta_sig / delta_sig_sum
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


"""
Replace a random row in the batch.
"""
def sample_minibatch_row(feed_dict, full_data, num_samples, B):
    i = np.random.randint(B)
    j = np.random.randint(num_samples)
    for (var, arr) in full_data.items():
        if len(arr.shape) > 1:
            feed_dict[var][i,:] = arr[j,:]
        else:
            feed_dict[var][i] = arr[j]


def estimate_tsne(
        full_data, vars, x0,
        num_pca_components, B, use_neural_network, sess,
        target_perplexity=30.0,
        # target_perplexity=50.0,
        alpha=1.0, use_vlr=False):

    # assume we don't have to reinitialize anything here
    feed_dict = dict()

    num_samples, n = np.shape(x0)

    target_perplexity = min(target_perplexity, float(num_samples)-1)

    tsne_all_sigmas = find_sigmas(x0, target_perplexity, use_vlr)
    print(tsne_all_sigmas)

    # tsne_all_sigmas = find_sigmas(x_loc_full, target_perplexity)

    # x_loc = tf.placeholder(tf.float32, (None, n), name="x_loc")
    # x_scale = tf.placeholder(tf.float32, (None, n), name="x_scale")
    # tsne_sigmas = tf.placeholder(
    #     tf.float32, shape=(B,), name="tsne_sigmas_batch")
    tsne_sigmas = tf.placeholder(
        tf.float32, shape=(None,), name="tsne_sigmas_batch")

    full_data[tsne_sigmas] = tsne_all_sigmas

    z0_mu = tf.placeholder(tf.float32, shape=(None, n-1), name="z0_mu")
    exp_x = rnaseq_approx_likelihood_sampler(
        z0_mu,
        efflens=vars["efflen"],
        mu=vars["la_mu"],
        sigma=vars["la_sigma"],
        alpha=vars["la_alpha"],
        left_index=vars["left_index"],
        right_index=vars["right_index"],
        leaf_index=vars["leaf_index"])
    full_data[z0_mu] = np.zeros((num_samples, n-1), dtype=np.float32)

    x = tf.log(exp_x)

    # using point estimates
    # x = tf.placeholder(tf.float32, shape=(None, n), name="x")
    # full_data[x] = x0

    # x = tf.log(
    #     rnaseq_approx_likelihood_sampler_from_vars(None, n, vars))

    tf.summary.histogram("x", x)

    # feed-forward network mapping onto low-dimensional latent space
    if use_neural_network:
        activation=tf.nn.leaky_relu

        lyr1 = tf.layers.dense(
            x, 500,
            activation=activation,
            # bias_initializer=tf.zeros_initializer(),
            kernel_initializer=tf.random_normal_initializer(0.0, 0.01))

        lyr2 = tf.layers.dense(
            lyr1, 500,
            activation=activation,
            # bias_initializer=tf.zeros_initializer())
            kernel_initializer=tf.random_normal_initializer(0.0, 0.01))

        # lyr3 = tf.layers.dense(
        #     lyr1, 2000,
        #     activation=activation,
        #     bias_initializer=tf.zeros_initializer())
        #     # kernel_initializer=tf.random_normal_initializer(0.0, 0.01))

        z = tf.layers.dense(
            lyr2,
            num_pca_components,
            kernel_initializer=tf.random_normal_initializer(0.0, 0.01),
            activation=tf.identity)
    else:
        z = tf.layers.dense(
            x,
            num_pca_components,
            activation=tf.identity,
            bias_initializer=tf.zeros_initializer(),
            kernel_initializer=tf.random_normal_initializer(0.0, 0.0001))

        # directly placing z (doesn't work with minibatch sampling)
        # z = tf.Variable(
        #     tf.random_normal([num_samples, num_pca_components]),
        #     trainable=True,
        #     name="z")

    tf.summary.histogram("z", z)

    # for numerical stability
    eps = 1e-6

    print(x)
    print(tsne_sigmas)

    p = tsne_p(x, tsne_sigmas, use_vlr) # [B, B]
    p += eps
    # p = tf.Print(p, [p], "p", summarize=25)

    q = tsne_q(z, alpha, use_vlr) # [B, B]
    q += eps

    tf.summary.histogram("p", p)
    tf.summary.histogram("q", q)
    tf.summary.histogram("log(p/q)", tf.log(p/q))
    tf.summary.histogram("p*log(p/q)", p*tf.log(p/q))

    # KL divergence between p and q
    loss = tf.reduce_sum(p * tf.log(p / q))
    # loss = tf.Print(loss, [tf.log(p/q)], "log(p/q)", summarize=16)
    # loss = tf.Print(loss, [p*tf.log(p/q)], "p*log(p/q)", summarize=16)
    # loss = tf.Print(loss, [loss], "loss")

    tf.summary.scalar("loss", loss)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=1e-6, beta1=0.9, beta2=0.99)
    train = optimizer.minimize(loss)

    # sample_minibatch(feed_dict, full_data, num_samples, B)
    feed_dict = full_data

    init = tf.global_variables_initializer()
    sess.run(init, feed_dict=feed_dict)

    # batch_feed_dict = {}
    # for var in full_data:
    #     batch_feed_dict[var] = None
    # n_iter = 5000
    n_iter = 50
    n_iter_per_batch = 10
    prog = Progbar(50, n_iter * n_iter_per_batch)
    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("log/" + "run-" + str(np.random.randint(1, 1000000)), sess.graph)
    k = 0
    for iter in range(n_iter):
        # sample_minibatch(feed_dict, full_data, num_samples, B)
        for subiter in range(n_iter_per_batch):
            # sample_minibatch_row(feed_dict, full_data, num_samples, B)
            _, loss_value = sess.run([train, loss], feed_dict=feed_dict)
            prog.update(k+1, loss=loss_value)
            train_writer.add_summary(
                sess.run(merged_summary, feed_dict=feed_dict), k)
            k += 1

    # sample z and average
    z_estimate = np.zeros((num_samples, num_pca_components))
    n_est_iterations = 10
    for iter in range(n_est_iterations):
        z_estimate += sess.run(z, feed_dict=full_data) / n_est_iterations

    # return z_estimate, sess.run(p, feed_dict=full_data), sess.run(q, feed_dict=full_data)
    return z_estimate

