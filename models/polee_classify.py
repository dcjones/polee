
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from polee_approx_likelihood import *
from polee_training import *
import polee_regression
import sys



class RNASeqLogisticRegression:
    def __init__(self, k, n):
        self.k = k
        self.encoder = tf.keras.Sequential()

        self.x_bias = tf.Variable(tf.zeros([n]))
        self.z_bias = tf.Variable(tf.zeros([k]))
        self.w = tf.Variable(tf.zeros([n, k]))

    def loss(self, x, z_true, loss_scale=1.0):
        z_logits = tf.matmul(x - self.x_bias, self.w) + self.z_bias

        loss = loss_scale * tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(z_true, logits=z_logits))

        # penalty = 0.0001
        # loss += penalty * tf.reduce_sum(tf.square(self.w))
        penalty = 0.001
        loss += penalty * tf.reduce_sum(tf.abs(self.w))

        # tf.print("w span", tf.reduce_min(self.w), tf.reduce_max(self.w))
        # tf.print("x bias", tf.reduce_min(self.x_bias), tf.reduce_max(self.x_bias))

        # penalty = 0.0001
        # loss += penalty * tf.reduce_sum(tf.abs(self.w))

        # tf.print("x", x)
        # tf.print("z_logits", z_logits)

        return loss

    def loss_sample(self, num_samples, n, vars, z_true, nsamples):
        loss = 0.0
        for _ in range(nsamples):
            x = tf.math.log(rnaseq_approx_likelihood_sampler_from_vars(num_samples, n, vars))
            loss += self.loss(x, z_true)
        loss /= nsamples
        return loss

    def fit_sample(self, num_samples, n, vars, z_true, niter, samples_per_iter=5):
        self.x_bias.assign(
            tf.reduce_mean(
                tf.math.log(rnaseq_approx_likelihood_sampler_from_vars(num_samples, n, vars)),
                axis=0))

        step_num = tf.Variable(1, trainable=False)

        @tf.function
        def trace_fn(loss, grad, vars):
            if tf.math.mod(step_num, 200) == 0:
                tf.print("[", step_num, "/", niter, "]  loss: ", loss, sep='')
            step_num.assign(step_num + 1)
            return loss

        tfp.math.minimize(
            loss_fn=lambda: self.loss_sample(num_samples, n, vars, z_true, samples_per_iter),
            num_steps=niter,
            optimizer=tf.optimizers.Adam(learning_rate=1e-4),
            trace_fn=trace_fn)

        return self.w.numpy()

    def fit(self, x, z_true, niter, loss_scale=1.0):
        self.x_bias.assign(tf.reduce_mean(x, axis=0))

        print(x.shape)
        tf.print("x", x)

        step_num = tf.Variable(1, trainable=False)

        @tf.function
        def trace_fn(loss, grad, vars):
            if tf.math.mod(step_num, 200) == 0:
                tf.print("[", step_num, "/", niter, "]  loss: ", loss, sep='')
            step_num.assign(step_num + 1)
            return loss

        tfp.math.minimize(
            loss_fn=lambda: self.loss(x, z_true, loss_scale),
            num_steps=niter,
            optimizer=tf.optimizers.Adam(learning_rate=1e-4),
            trace_fn=trace_fn)

        return self.w.numpy()

    def eval_sample(self, num_samples, n, vars):
        x = tf.math.log(rnaseq_approx_likelihood_sampler_from_vars(num_samples, n, vars))
        return self.eval(x)

    def eval(self, x):
        z_logits = tf.matmul(x - self.x_bias, self.w) + self.z_bias
        return tf.nn.softmax(z_logits, axis=-1)

    def predict_sample(self, num_samples, n, vars, niter):
        y_probs = tf.Variable(tf.zeros([num_samples, self.k]))
        for i in range(niter):
            y_probs.assign_add(
                self.eval_sample(num_samples, n, vars))

        return (y_probs / niter).numpy()

    def predict(self, x):
        return self.eval(x).numpy()
