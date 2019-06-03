
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from polee_approx_likelihood import *
from polee_training import *
import sys



def classification_model(x, vars, n, K, num_samples, dropout_rate):

    lyr1 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
    lyr1_output = lyr1(x)

    drp1 = tf.keras.layers.Dropout(rate=dropout_rate)
    drp1_output = drp1(lyr1_output)

    lyr2 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
    lyr2_output = lyr2(drp1_output)

    drp2 = tf.keras.layers.Dropout(rate=dropout_rate)
    drp2_output = drp2(lyr2_output)

    lyrn = tf.keras.layers.Dense(K)
    z_predict_logits = lyrn(drp2_output)

    return z_predict_logits, lyr1, lyr2, lyrn

    # regularizer = tf.keras.regularizers.l2(0.01)

    # lyr1 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
    # lyr1_output = lyr1(x)

    # lyr2 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
    # lyr2_output = lyr2(lyr1_output)

    # lyr3 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
    # lyr3_output = lyr3(lyr2_output)

    # lyrn = tf.keras.layers.Dense(K)
    # z_predict_logits = lyrn(lyr3_output)

    # z_predict_logits = lyrn(x)

    # return z_predict_logits, lyr1, lyr2, lyr3, lyrn


# K is number of components, D is dimensionality of the latent space
def train_classifier(
        sess, init_feed_dict, num_samples, n, vars, x_init, z_true,
        use_posterior_mean):

    K = z_true.shape[1] # number of categories

    if use_posterior_mean:
        x = tf.constant(x_init, name="x")
    else:
        # x = rnaseq_approx_likelihood_sampler_from_vars(num_samples, n, vars)

        # Seems to do much worse on log scale
        x = tf.log(rnaseq_approx_likelihood_sampler_from_vars(num_samples, n, vars))

    # z_predict_logits, lyr1, lyr2, lyr3, lyrn = classification_model(x, vars, n, K, num_samples, 0.5)
    z_predict_logits, lyr1, lyr2, lyrn = classification_model(x, vars, n, K, num_samples, 0.5)

    # z_predict_logits = tf.Print(
    #     z_predict_logits, [tf.reduce_min(z_predict_logits), tf.reduce_max(z_predict_logits)], "z_predict_logits span")

    loss = tf.losses.softmax_cross_entropy(z_true, logits=z_predict_logits)

    # train(sess, loss, init_feed_dict, 2000, 1e-3)
    train(sess, loss, init_feed_dict, 3000, 1e-3)

    lyr1_weights = sess.run(lyr1.weights)
    lyr2_weights = sess.run(lyr2.weights)
    # lyr3_weights = sess.run(lyr3.weights)
    lyrn_weights = sess.run(lyrn.weights)

    return {
        "lyr1": lyr1_weights,
        "lyr2": lyr2_weights,
        # "lyr3": lyr3_weights,
        "lyrn": lyrn_weights
    }


def run_classifier(
        sess, classify_model, init_feed_dict, num_samples, n, vars, x_init, K,
        use_posterior_mean):

    if use_posterior_mean:
        x = tf.constant(x_init, name="x")
    else:
        # x = rnaseq_approx_likelihood_sampler_from_vars(num_samples, n, vars)

        x = tf.log(rnaseq_approx_likelihood_sampler_from_vars(num_samples, n, vars))

    # z_predict_logits, lyr1, lyr2, lyr3, lyrn = classification_model(x, vars, n, K, num_samples, 0.0)
    z_predict_logits, lyr1, lyr2, lyrn = classification_model(x, vars, n, K, num_samples, 0.0)

    # z_predict_logits = tf.Print(
    #     z_predict_logits, [tf.reduce_min(z_predict_logits), tf.reduce_max(z_predict_logits)], "z_predict_logits span")

    # Idea here is to run this a bunch of times and average the results
    init = tf.global_variables_initializer()
    sess.run(init, feed_dict=init_feed_dict)

    # lyr1.set_weights(classify_model["lyr1"])
    # lyrn.set_weights(classify_model["lyrn"])
    sess.run(tf.assign(lyr1.weights[0], classify_model["lyr1"][0]))
    sess.run(tf.assign(lyr1.weights[1], classify_model["lyr1"][1]))
    sess.run(tf.assign(lyr2.weights[0], classify_model["lyr2"][0]))
    sess.run(tf.assign(lyr2.weights[1], classify_model["lyr2"][1]))
    # sess.run(tf.assign(lyr3.weights[0], classify_model["lyr3"][0]))
    # sess.run(tf.assign(lyr3.weights[1], classify_model["lyr3"][1]))
    sess.run(tf.assign(lyrn.weights[0], classify_model["lyrn"][0]))
    sess.run(tf.assign(lyrn.weights[1], classify_model["lyrn"][1]))

    n_iter = 50
    prog = Progbar(50, n_iter)
    z_predict_logits_mean = np.zeros((num_samples, K), np.float32)
    for iter in range(n_iter):
        z_predict_logits_mean += sess.run(z_predict_logits)
        prog.update(iter, loss=0.0)
    z_predict_logits_mean /= n_iter

    return sess.run(tf.nn.softmax(z_predict_logits_mean, axis=1))

