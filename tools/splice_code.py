
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sys
from tensorflow_probability import distributions as tfd

def estimate_splicing_code(
        qx_feature_loc, qx_feature_scale,
        donor_seqs, acceptor_seqs, alt_donor_seqs, alt_acceptor_seqs, tissues):

    num_samples = len(tissues)
    num_tissues = np.max(tissues)

    print(tissues)
    print(num_samples)
    print(num_tissues)

    tissue_matrix = np.zeros((num_samples, num_tissues), dtype=np.float32)
    for (i, j) in enumerate(tissues):
        tissue_matrix[i, j-1] = 1

    seqs = np.hstack(
        [donor_seqs, acceptor_seqs, alt_donor_seqs, alt_acceptor_seqs])
        # [ num_features, seq_length, 4 ]

    # TODO: set this up for testing
    # To do so, I think we need to use a placeholder, split seqs randomly
    # then go from there.

    # lyr1_input_init = tf.placeholder(tf.float32, (None, seqs.shape[1], seqs.shape[2]))
    # lyr1_input = tf.Variable(
    #     lyr1_input_init,
    #     trainable=False,
    #     name="lyr1_input")

    # TODO: this is maybe just too dense. What can be done?
    lyr1 = tf.layers.dense(
        inputs=tf.constant(seqs),
        units=64,
        activation=tf.nn.relu)

    lyr2 = tf.layers.dense(
        inputs=tf.layers.flatten(lyr1),
        units=num_tissues,
        activation=tf.nn.sigmoid)
        # [num_features, num_tissues]

    # TODO: eventually this should be a latent variable
    x_scale = 0.05

    # TODO: we have to kind of cast x_mu
    # onto the right dimensionality

    x_mu = tf.matmul(
        tf.constant(tissue_matrix),
        tf.transpose(lyr2))
        # [num_samples, num_features]

    print(x_mu)

    x_prior = tfd.Normal(
        loc=x_mu,
        scale=x_scale,
        name="x_prior")

    x = tf.Variable(
        qx_feature_loc,
        name="x")

    x_likelihood = tfd.Normal(
        loc=qx_feature_loc,
        scale=qx_feature_scale,
        name="x_likelihood")

    log_posterior = \
        tf.reduce_sum(x_prior.log_prob(x)) + tf.reduce_sum(x_likelihood.log_prob(x))

    sess = tf.Session()

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
    train = optimizer.minimize(-log_posterior)

    # sess.run(
    #     tf.global_variables_initializer(),
    #     feed_dict={lyr1_input_init: seqs})
    sess.run(tf.global_variables_initializer())

    n_iter = 100
    for iter in range(n_iter):
        _, log_posterior_value = sess.run([train, log_posterior])
        print(log_posterior_value)
