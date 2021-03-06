
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sys


class Progbar:
    def __init__(self, width, n_iter):
        self.width = width
        self.n_iter = n_iter
        self.curr_text = ""

    def update(self, iter, loss):
        # delete current text
        sys.stdout.write("\b" * len(self.curr_text))
        sys.stdout.write("\r")

        # write new text
        num_filled_bars = round(self.width * iter / self.n_iter)
        num_blank_bars = self.width - num_filled_bars
        progbar = "["
        progbar += "=" * num_filled_bars
        progbar += " " * num_blank_bars
        progbar += "] LOSS: {loss:0.1f}".format(loss=loss)

        sys.stdout.write(progbar)
        sys.stdout.flush()
        self.curr_text = progbar


# # From: https://github.com/tensorflow/probability/issues/100
# class HalfCauchy(tfp.distributions.TransformedDistribution):
#   def __init__(self, loc, scale, name, validate_args=False):
#     super(HalfCauchy, self).__init__(
#         distribution=tfp.distributions.Cauchy(loc, scale, validate_args=validate_args),
#         bijector=tfp.bijectors.AbsoluteValue(validate_args=validate_args),
#         name=name)


def train(
        sess, objective, init_feed_dict, n_iter, init_learning_rate,
        var_list=None, initialized_vars=None, write_logs=False,
        decay_rate=1.0, decay_boundaries=None, decay_values=None):

    if var_list is None:
        var_list = tf.trainable_variables()

    global_step = tf.Variable(0, trainable=False)

    if decay_boundaries is not None or decay_values is not None:
        assert decay_boundaries is not None
        assert decay_values is not None
        learning_rate = tf.train.piecewise_constant_decay(
            global_step, decay_boundaries, decay_values)
    else:
        learning_rate = tf.train.exponential_decay(
            init_learning_rate, global_step, 10, decay_rate)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    train = optimizer.minimize(objective, var_list=var_list, global_step=global_step)
    if initialized_vars is None:
        init = tf.global_variables_initializer()
    else:
        init = tf.initialize_variables(set(tf.global_variables()) - initialized_vars)
    prog = Progbar(50, n_iter)

    if write_logs:
        tf.summary.scalar("loss", objective)
        merged_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("log/" + "run-" + str(np.random.randint(1, 1000000)), sess.graph)

    sess.run(init, feed_dict=init_feed_dict)
    for iter in range(n_iter):
        sess.run(train)
        obj_value = sess.run(objective)
        prog.update(iter, loss=obj_value)
        if write_logs:
            train_writer.add_summary(sess.run(merged_summary), iter)
    print()


