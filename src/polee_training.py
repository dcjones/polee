
import tensorflow as tf
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


def train(sess, objective, init_feed_dict, n_iter, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(objective)
    init = tf.global_variables_initializer()
    prog = Progbar(50, n_iter)

    tf.summary.scalar("loss", objective)
    merged_summary = tf.summary.merge_all()

    # with tf.Session() as sess:
    train_writer = tf.summary.FileWriter("log/" + "run-" + str(np.random.randint(1, 1000000)), sess.graph)
    sess.run(init, feed_dict=init_feed_dict)
    for iter in range(n_iter):
        sess.run(train)
        obj_value = sess.run(objective)
        prog.update(iter, loss=obj_value)
        train_writer.add_summary(sess.run(merged_summary), iter)


