
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sys
from tensorflow_probability import distributions as tfd
from tensorflow_probability import edward2 as ed


def median_absolute_deviance_sample(x_mu, x_likelihood):
    x_mu_sigmoid = tf.sigmoid(x_mu)
    x_sigmoid = tf.sigmoid(x_likelihood)
    return tfp.distributions.percentile(tf.abs(x_mu_sigmoid - x_sigmoid), 50.0)


def est_expected_median_absolute_deviance(sess, mad_sample, feed_dict):
    n_iter = 50
    return sum(sess.run(mad_sample, feed_dict=feed_dict) for iter in range(n_iter)) / n_iter


def estimate_splicing_code(
        qx_feature_loc, qx_feature_scale,
        donor_seqs, acceptor_seqs, alt_donor_seqs, alt_acceptor_seqs, tissues):

    num_samples = len(tissues)
    num_tissues = np.max(tissues)

    tissue_matrix = np.zeros((num_samples, num_tissues), dtype=np.float32)
    for (i, j) in enumerate(tissues):
        tissue_matrix[i, j-1] = 1

    seqs = np.hstack(
        [donor_seqs, acceptor_seqs, alt_donor_seqs, alt_acceptor_seqs])
        # [ num_features, seq_length, 4 ]
    num_features = seqs.shape[0]

    # split into testing and training data
    shuffled_feature_idxs = np.arange(num_features)
    np.random.shuffle(shuffled_feature_idxs)

    seqs_train_len = int(np.floor(0.75 * num_features))
    seqs_test_len  = num_features - seqs_train_len

    train_idxs = shuffled_feature_idxs[:seqs_train_len]
    test_idxs = shuffled_feature_idxs[seqs_train_len:]

    seqs_train = seqs[train_idxs]
    seqs_test = seqs[test_idxs]

    qx_feature_loc_train = qx_feature_loc[:,train_idxs]
    qx_feature_scale_train = qx_feature_scale[:,train_idxs]

    qx_feature_loc_test = qx_feature_loc[:,test_idxs]
    qx_feature_scale_test = qx_feature_scale[:,test_idxs]

    keep_prob = tf.placeholder(tf.float32)

    # model
    lyr1_input = tf.placeholder(tf.float32, (None, seqs.shape[1], seqs.shape[2]))

    lyr0 = tf.nn.dropout(tf.layers.flatten(lyr1_input), keep_prob)

    lyr1 = tf.layers.dense(
        inputs=lyr0,
        units=200,
        # kernel_regularizer=tf.contrib.layers.l1_regularizer(0.1),
        activation=tf.nn.tanh)

    lyr1_dropout = tf.nn.dropout(lyr1, keep_prob)

    lyr2 = tf.layers.dense(
        inputs=lyr1_dropout,
        units=200,
        # kernel_regularizer=tf.contrib.layers.l1_regularizer(0.1),
        activation=tf.nn.relu)

    lyr2_dropout = tf.nn.dropout(lyr2, keep_prob)

    lyr3 = tf.layers.dense(
        inputs=lyr2_dropout,
        units=200,
        activation=tf.nn.relu)

    lyr3_dropout = tf.nn.dropout(lyr3, keep_prob)

    lyr4 = tf.layers.dense(
        inputs=lyr1_dropout,
        units=num_tissues,
        # kernel_regularizer=tf.contrib.layers.l1_regularizer(0.1),
        activation=tf.identity)
        # [num_features, num_tissues]

    # TODO: eventually this should be a latent variable
    x_scale = 0.01

    x_mu = tf.matmul(
        tf.constant(tissue_matrix),
        tf.transpose(lyr4))
        # [num_samples, num_features]

    x_prior = tfd.Normal(
        loc=x_mu,
        scale=x_scale,
        name="x_prior")

    x_likelihood_loc = tf.placeholder(tf.float32, [num_samples, None])
    x_likelihood_scale = tf.placeholder(tf.float32, [num_samples, None])
    x_likelihood = ed.Normal(
        loc=x_likelihood_loc,
        scale=x_likelihood_scale,
        name="x_likelihood")

    # x = tf.Variable(
    #     qx_feature_loc_train,
    #     name="x")

    # print(x_prior.log_prob(x))
    # print(x_likelihood.log_prob(x))
    # sys.exit()

    # log_prior = tf.reduce_sum(x_prior.log_prob(x))
    # log_likelihood = tf.reduce_sum(x_likelihood.distribution.log_prob(x))

    # log_posterior = log_prior + log_likelihood

    log_posterior = x_likelihood.distribution.log_prob(x_mu)

    sess = tf.Session()

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
    train = optimizer.minimize(-log_posterior)

    sess.run(tf.global_variables_initializer())

    train_feed_dict = {
        keep_prob: 0.9,
        lyr1_input: seqs_train,
        x_likelihood_loc: qx_feature_loc_train,
        x_likelihood_scale: qx_feature_scale_train }

    test_feed_dict = {
        keep_prob: 1.0,
        lyr1_input: seqs_test,
        x_likelihood_loc: qx_feature_loc_test,
        x_likelihood_scale: qx_feature_scale_test }

    n_iter = 2000
    mad_sample = median_absolute_deviance_sample(x_mu, x_likelihood)
    for iter in range(n_iter):
        # _, log_prior_value, log_likelihood_value = sess.run(
        #     [train, log_prior, log_likelihood],
        #     feed_dict=train_feed_dict)

        sess.run(
            [train],
            feed_dict=train_feed_dict)

        # print((log_prior_value, log_likelihood_value))

        if iter % 100 == 0:
            print(iter)
            print(est_expected_median_absolute_deviance(sess, mad_sample, train_feed_dict))
            print(est_expected_median_absolute_deviance(sess, mad_sample, test_feed_dict))

    print(est_expected_median_absolute_deviance(sess, mad_sample, train_feed_dict))
    print(est_expected_median_absolute_deviance(sess, mad_sample, test_feed_dict))

    # TODO: test accuracy, somehow

    # I guess we basically want to compare
    # x_prior against the likelihood

    # kl_pq, kl_qp, avg_log_lik = sess.run(
    #     [tf.reduce_mean(x_prior.kl_divergence(x_likelihood)),
    #      tf.reduce_mean(x_likelihood.kl_divergence(x_prior)),
    #      tf.reduce_mean(x_prior.log_prob(x_likelihood_loc))],
    #     feed_dict={
    #         keep_prob: 1.0,
    #         lyr1_input: seqs_train,
    #         x_likelihood_loc: qx_feature_loc_train,
    #         x_likelihood_scale: qx_feature_scale_train })

    # print((avg_log_lik, kl_qp, kl_pq, (kl_qp + kl_pq)/2))


    # kl_pq, kl_qp, avg_log_lik = sess.run(
    #     [tf.reduce_mean(x_prior.kl_divergence(x_likelihood)),
    #      tf.reduce_mean(x_likelihood.kl_divergence(x_prior)),
    #      tf.reduce_mean(x_prior.log_prob(x_likelihood_loc))],
    #     feed_dict={
    #         keep_prob: 1.0,
    #         lyr1_input: seqs_test,
    #         x_likelihood_loc: qx_feature_loc_test,
    #         x_likelihood_scale: qx_feature_scale_test })

    # print((avg_log_lik, kl_qp, kl_pq, (kl_qp + kl_pq)/2))

