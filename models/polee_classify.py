
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from polee_approx_likelihood import *
from polee_training import *
import sys


# split data into training and test sets
def make_train_test_sets(
        full_data, x_init, z_true,
        num_samples, num_train_samples):
    idxs = np.array([i for i in range(num_samples)])
    np.random.shuffle(idxs)

    train_idx = idxs[0:num_train_samples]
    test_idx = idxs[num_train_samples:]

    train_feed_dict = {}
    test_feed_dict = {}

    for (var, arr) in full_data.items():
        if len(arr.shape) > 1:
            train_feed_dict[var] = arr[train_idx,:]
            test_feed_dict[var] = arr[test_idx,:]
        else:
            train_feed_dict[var] = arr[train_idx]
            test_feed_dict[var] = arr[test_idx]

    # train_feed_dict[z_ph] = z_true[train_idx,:]
    # test_feed_dict[z_ph] = tf.fill([num_samples-num_train_samples, K], 1/K)

    # train_feed_dict[x_ph] = x_init[train_idx,:]
    # test_feed_dict[x_ph] = x_init[test_idx,:]

    return (train_feed_dict, test_feed_dict, train_idx, test_idx)



def classification_model(
        vars, n, K, num_samples, training=True,
        temperature_init=5.0, p_logits_init=None):

    if p_logits_init is None:
        p_logits_init = tf.zeros(K)

    log_prior = 0.0

    # TODO: figure out how to fit this if we can't use gumbel-softmax during
    # training.
    p_logits = tf.Variable(p_logits_init, name="p_logits", trainable=False)
    p = tf.nn.softmax(p_logits)

    # p_prior = tfd.Dirichlet(tf.ones(K))
    # log_prior += tf.reduce_sum(p_prior.log_prob(p))

    if training:
        z_ph = tf.placeholder(tf.float32, shape=(num_samples, K))
        z = tf.Variable(z_ph, name="z", trainable=False)
    else:
        z_ph = None
        z = tf.nn.softmax(tf.Variable(tf.zeros((num_samples, K)), name="z"), axis=1)
        # z = tf.Variable(tf.fill((num_samples, K), 1/K), name="z")

        temperature = tf.Variable(temperature_init, name="temperature")
        z_prior = tfd.RelaxedOneHotCategorical(
            temperature,
            probs=p,
            name="z_prior")
        # log_prior += tf.reduce_sum(z_prior.log_prob(z))

    lyr1 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu, use_bias=True)
    lyr1_output = lyr1(z)
    lyrn = tf.keras.layers.Dense(n)
    lyrn_output = lyrn(lyr1_output)
    x_loc = lyrn_output

    x_scale = tf.nn.softplus(tf.Variable(tf.fill([n], -3.0), name="x_scale"))
    x_scale_prior = tfd.HalfCauchy(
        loc=0.0,
        scale=0.1,
        name="x_scale_prior")
    log_prior += tf.reduce_sum(x_scale_prior.log_prob(x_scale))

    x_prior = tfd.StudentT(
        loc=x_loc,
        scale=x_scale,
        df=1.0,
        name="x_prior")

    x_ph = tf.placeholder(tf.float32, shape=(num_samples, n))
    x = tf.Variable(x_ph, name="x")
    log_prior += tf.reduce_sum(x_prior.log_prob(x))

    log_likelihood = rnaseq_approx_likelihood_from_vars(vars, x)

    log_posterior = log_likelihood + log_prior

    return log_posterior, z, z_ph, x_ph, lyr1, lyrn


# K is number of components, D is dimensionality of the latent space
def train_classifier(
        sess, init_feed_dict, num_samples, n, vars, x_init, z_true):

    K = z_true.shape[1] # number of categories

    log_posterior, z, z_ph, x_ph, lyr1, lyrn = \
        classification_model(vars, n, K, num_samples, training=True)

    init_feed_dict[z_ph] = z_true
    init_feed_dict[x_ph] = x_init

    tf.summary.scalar("log_posterior", log_posterior)

    learning_rate = 2e-2
    n_iter = 500

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(-log_posterior, var_list=tf.trainable_variables())

    prog = Progbar(50, n_iter)
    merged_summary = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer(), feed_dict=init_feed_dict)

    train_writer = tf.summary.FileWriter("log/" + "run-" + str(np.random.randint(1, 1000000)), sess.graph)
    for iter in range(n_iter):
        _, obj_value = sess.run([train, -log_posterior])
        prog.update(iter, loss=obj_value)
        train_writer.add_summary(sess.run(merged_summary), iter)
    print()

    lyr1_weights = lyr1.get_weights()
    lyrn_weights = lyrn.get_weights()
    print(lyr1_weights)
    print(lyrn_weights)

    return {
        "lyr1_weights": lyr1_weights,
        "lyrn_weights": lyrn_weights
    }


def run_classifier(
        sess, classify_model, init_feed_dict, num_samples, n, vars, x_init, K):

    log_posterior, z, z_ph, x_ph, lyr1, lyrn = \
        classification_model(vars, n, K, num_samples, training=False)

    init_feed_dict[x_ph] = x_init

    tf.summary.scalar("log_posterior", log_posterior)

    learning_rate = 2e-2
    n_iter = 500

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(-log_posterior, var_list=tf.trainable_variables())

    prog = Progbar(50, n_iter)

    sess.run(tf.global_variables_initializer(), feed_dict=init_feed_dict)

    lyr1.set_weights(classify_model["lyr1_weights"])
    lyrn.set_weights(classify_model["lyrn_weights"])

    for iter in range(n_iter):
        _, obj_value = sess.run([train, -log_posterior])
        prog.update(iter, loss=obj_value)
    print()

    return sess.run(z)

