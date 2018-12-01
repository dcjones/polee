
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

    # x_sigmoid = tf.sigmoid(x_likelihood.distribution.loc)
    # return tf.reduce_mean(tf.square(x_mu_sigmoid - x_sigmoid))


def est_expected_median_absolute_deviance(sess, mad_sample, feed_dict):
    n_iter = 5
    return sum(sess.run(mad_sample, feed_dict=feed_dict) for iter in range(n_iter)) / n_iter


def estimate_splicing_code(
        qx_feature_loc, qx_feature_scale,
        donor_seqs, acceptor_seqs, alt_donor_seqs, alt_acceptor_seqs,
        donor_cons, acceptor_cons, alt_donor_cons, alt_acceptor_cons,
        tissues):

    num_samples = len(tissues)
    num_tissues = np.max(tissues)

    tissue_matrix = np.zeros((num_samples, num_tissues), dtype=np.float32)
    for (i, j) in enumerate(tissues):
        tissue_matrix[i, j-1] = 1

    seqs = np.hstack(
        [donor_seqs, acceptor_seqs, alt_donor_seqs, alt_acceptor_seqs])
        # [ num_features, seq_length, 4 ]

    cons = np.hstack(
        [donor_cons, acceptor_cons, alt_donor_cons, alt_acceptor_cons])

    seqs = np.concatenate((seqs, np.expand_dims(cons, 2)), axis=2)
    print(seqs.shape)

    # sys.exit()

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
    lyr0_input = tf.placeholder(tf.float32, (None, seqs.shape[1], seqs.shape[2]))
    # lyr0 = tf.layers.flatten(lyr0_input)
    lyr0 = lyr0_input

    print(lyr0)

    conv1 = tf.layers.conv1d(
        inputs=lyr0,
        filters=64,
        kernel_size=4,
        activation=tf.nn.leaky_relu,
        name="conv1")

    print(conv1)

    pool1 = tf.layers.max_pooling1d(
        inputs=conv1,
        pool_size=2,
        strides=2,
        name="pool1")

    print(pool1)

    conv2 = tf.layers.conv1d(
        inputs=pool1,
        filters=64,
        kernel_size=4,
        activation=tf.nn.leaky_relu,
        name="conv2")

    pool2 = tf.layers.max_pooling1d(
        inputs=conv2,
        pool_size=2,
        strides=2,
        name="pool2")

    pool2_flat = tf.layers.flatten(
        pool2, name="pool2_flat")

    dense1 = tf.layers.dense(
        inputs=pool2_flat,
        units=256,
        activation=tf.nn.leaky_relu,
        name="dense1")

    training_flag = tf.placeholder(tf.bool)

    dropout1 = tf.layers.dropout(
        inputs=dense1,
        rate=0.4,
        training=training_flag,
        name="dropout1")

    prediction_layer = tf.layers.dense(
        inputs=dropout1,
        units=num_tissues,
        activation=tf.identity)
        # [num_features, num_tissues]

    # TODO: eventually this should be a latent variable
    x_scale = 0.1

    x_mu = tf.matmul(
        tf.constant(tissue_matrix),
        tf.transpose(prediction_layer))
        # [num_samples, num_features]

    x_prior = tfd.Normal(
        loc=x_mu,
        # loc=0.0,
        scale=x_scale,
        name="x_prior")

    # x_prior = tfd.StudentT(
    #     loc=x_mu,
    #     scale=x_scale,
    #     df=2.0,
    #     name="x_prior")

    x_likelihood_loc = tf.placeholder(tf.float32, [num_samples, None])
    x_likelihood_scale = tf.placeholder(tf.float32, [num_samples, None])
    x_likelihood = ed.Normal(
        loc=x_likelihood_loc,
        scale=x_likelihood_scale,
        name="x_likelihood")

    # x = x_likelihood

    x = tf.Variable(
        # qx_feature_loc_train,
        # tf.random_normal(qx_feature_loc_train.shape),
        # tf.zeros(qx_feature_loc_train.shape),
        qx_feature_loc_train + qx_feature_scale_train * np.float32(np.random.randn(*qx_feature_loc_train.shape)),
        # trainable=False,
        name="x")

    # x_delta = tf.Variable(
    #     # qx_feature_loc_train,
    #     # tf.random_normal(qx_feature_loc_train.shape),
    #     tf.zeros(qx_feature_loc_train.shape),
    #     # trainable=False,
    #     name="x")

    # x_delta = tf.Print(x_delta,
    #     [tf.reduce_min(x_delta), tf.reduce_max(x_delta)], "x_delta span")

    # x = tf.Print(x,
    #     [tf.reduce_min(x - qx_feature_loc_train), tf.reduce_max(x - qx_feature_loc_train)],
    #     "x deviance from init")

    # print(x_prior.log_prob(x))
    # print(x_likelihood.log_prob(x))
    # sys.exit()

    # log_prior = tf.reduce_sum(x_prior.log_prob(x_delta))
    # log_likelihood = tf.reduce_sum(x_likelihood.distribution.log_prob(x_mu + x_delta))

    log_prior = tf.reduce_sum(x_prior.log_prob(x))
    log_likelihood = tf.reduce_sum(x_likelihood.distribution.log_prob(x))

    log_posterior = log_prior + log_likelihood

    # log_posterior = x_likelihood.distribution.log_prob(x_mu)

    sess = tf.Session()

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train = optimizer.minimize(-log_posterior)

    sess.run(tf.global_variables_initializer())

    # dropout doesn't seem to do much....
    train_feed_dict = {
        training_flag: True,
        # training_flag: False,
        lyr0_input: seqs_train,
        x_likelihood_loc: qx_feature_loc_train,
        x_likelihood_scale: qx_feature_scale_train }

    test_feed_dict = {
        training_flag: False,
        lyr0_input: seqs_test,
        x_likelihood_loc: qx_feature_loc_test,
        x_likelihood_scale: qx_feature_scale_test }

    n_iter = 10000
    mad_sample = median_absolute_deviance_sample(x_mu, x_likelihood)
    for iter in range(n_iter):
        _, log_prior_value, log_likelihood_value = sess.run(
            [train, log_prior, log_likelihood],
            feed_dict=train_feed_dict)

        # sess.run(
        #     [train],
        #     feed_dict=train_feed_dict)

        print((log_prior_value, log_likelihood_value))

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


def estimate_splicing_code_from_kmers(
        qx_feature_loc, qx_feature_scale, kmer_usage_matrix, tissues):

    num_samples = len(tissues)
    num_tissues = np.max(tissues)

    tissue_matrix = np.zeros((num_samples, num_tissues), dtype=np.float32)
    for (i, j) in enumerate(tissues):
        tissue_matrix[i, j-1] = 1

    num_features = kmer_usage_matrix.shape[0]
    num_kmers = kmer_usage_matrix.shape[1]

    # split into testing and training data
    shuffled_feature_idxs = np.arange(num_features)
    np.random.shuffle(shuffled_feature_idxs)

    seqs_train_len = int(np.floor(0.75 * num_features))
    seqs_test_len  = num_features - seqs_train_len

    train_idxs = shuffled_feature_idxs[:seqs_train_len]
    test_idxs = shuffled_feature_idxs[seqs_train_len:]

    kmer_usage_matrix_train = kmer_usage_matrix[train_idxs]
    kmer_usage_matrix_test = kmer_usage_matrix[test_idxs]

    qx_feature_loc_train = qx_feature_loc[:,train_idxs]
    qx_feature_scale_train = qx_feature_scale[:,train_idxs]

    qx_feature_loc_test = qx_feature_loc[:,test_idxs]
    qx_feature_scale_test = qx_feature_scale[:,test_idxs]

    W0 = tf.Variable(
        tf.random_normal([num_kmers, 1], mean=0.0, stddev=0.01),
        name="W0")

    # B = tf.Variable(
    #     tf.random_normal([1, num_tissues], mean=0.0, stddev=0.01),
    #     name="B")

    W_prior = tfd.Normal(
        loc=0.0,
        scale=0.1,
        name="W_prior")

    W = tf.Variable(
        tf.random_normal([num_kmers, num_tissues], mean=0.0, stddev=0.01),
        name="W")

    X = tf.placeholder(tf.float32, shape=(None, num_kmers), name="X")

    # Y = B + tf.matmul(X, W0 + W)
    Y = tf.matmul(X, W0 + W)

    print(Y)

    x_scale_prior = tfd.InverseGamma(
        concentration=0.001,
        rate=0.001,
        name="x_scale_prior")

    x_scale = tf.nn.softplus(tf.Variable(tf.fill([seqs_train_len], -3.0)))

    x_mu = tf.matmul(
        tf.constant(tissue_matrix),
        tf.transpose(Y))
        # [num_samples, num_features]

    print(x_mu)

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



    # Using likelihood

    x = tf.Variable(
        qx_feature_loc_train,
        name="x")

    # x = x_likelihood_loc

    # x = x_mu

    log_prior = \
        tf.reduce_sum(x_prior.log_prob(x)) + \
        tf.reduce_sum(x_scale_prior.log_prob(x_scale)) + \
        tf.reduce_sum(W_prior.log_prob(W))

    log_likelihood = tf.reduce_sum(x_likelihood.distribution.log_prob(x))

    log_posterior = log_prior + log_likelihood



    # Using point estimates

    # x = qx_feature_loc_train

    # log_prior = \
    #     tf.reduce_sum(x_prior.log_prob(x)) + \
    #     tf.reduce_sum(x_scale_prior.log_prob(x_scale)) + \
    #     tf.reduce_sum(W_prior.log_prob(W))

    # log_posterior = log_prior


    sess = tf.Session()

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    train = optimizer.minimize(-log_posterior)

    sess.run(tf.global_variables_initializer())

    train_feed_dict = {
        X: kmer_usage_matrix_train,
        x_likelihood_loc: qx_feature_loc_train,
        x_likelihood_scale: qx_feature_scale_train }

    test_feed_dict = {
        X: kmer_usage_matrix_test,
        x_likelihood_loc: qx_feature_loc_test,
        x_likelihood_scale: qx_feature_scale_test }

    n_iter = 1000
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
            print(sess.run(tf.reduce_min(x_scale)))
            print(sess.run(tf.reduce_max(x_scale)))
            # print(sess.run(log_prior, feed_dict=train_feed_dict))
            # print(sess.run(log_likelihood, feed_dict=train_feed_dict))

    return sess.run(W0), sess.run(W)