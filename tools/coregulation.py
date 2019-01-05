
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sys
import math
from tensorflow_probability import distributions as tfd
from tensorflow_probability import edward2 as ed

def fillmask(mask_init_value, start_j, batch_size):
    for (k, j) in enumerate(range(start_j, start_j+batch_size)):
        mask_init_value[:, k] = 0
        mask_init_value[j+1:, k] = 1


def estimate_gmm_precision(qx_loc, qx_scale, batch_size=1):
    num_samples = qx_loc.shape[0]
    n = qx_loc.shape[1]

    # [num_samples, n]
    # TODO: testing with fixed qx
    qx = ed.Normal(
        loc=qx_loc,
        scale=qx_scale,
        name="qx")

    qw_loc_init = tf.placeholder(tf.float32, (n, batch_size), name="qw_loc_init")
    qw_loc_init_value = np.zeros((n, batch_size), dtype=np.float32)
    qw_loc = tf.Variable(qw_loc_init, name="qw_loc")

    qw_scale_softminus_init = tf.placeholder(tf.float32, (n, batch_size), name="qw_scale_softminus_init")
    # TODO: seems very sensitive to how we initialize this. That's not such a great property.
    qw_scale_softminus_init_value = np.full((n,batch_size), -2.0, dtype=np.float32)
    qw_scale = tf.nn.softplus(tf.Variable(qw_scale_softminus_init, name="qw_scale"))

    # [n, batch_size]
    qw = ed.Normal(
        loc=qw_loc,
        scale=qw_scale,
        name="qw")

    # TODO: testing with no distribution
    # qw = qw_loc

    # TODO: maybe try something like horseshoe prior
    w_prior = tfd.Normal(
        loc=0.0,
        scale=0.01)

    err_scale = 0.1

    # [n, batch_size]
    mask_init = tf.placeholder(tf.float32, (n, batch_size), name="mask_init")
    mask_init_value = np.empty([n, batch_size], dtype=np.float32)
    mask = tf.Variable(mask_init, name="mask", trainable=False)

    x_mean, x_var = tf.nn.moments(qx, axes=0) # [n]
    # x_var = tf.Print(x_var, [tf.reduce_min(x_var), tf.reduce_mean(x_var), tf.reduce_max(x_var)], "x_var")

    x_std = (qx - x_mean) / tf.sqrt(x_var) # [num_samples, n]
    # x_std = qx
    # x_std = tf.Print(x_std, [tf.reduce_min(x_std), tf.reduce_mean(x_std), tf.reduce_max(x_std)], "x_std")

    qw_masked = qw * mask
    # qw_masked = qw

    # [num_samples, batch_size]
    y_dist = tfd.Normal(
        loc=tf.matmul(x_std, qw_masked),
        scale=err_scale)

    # TODO:
    y_slice_start_init = tf.placeholder(tf.int32, 2, name="y_slice_start_init") # set to [0, j]
    y_slice_start = tf.Variable(y_slice_start_init, name="y_slice_start", trainable=False)
    y = tf.slice(x_std, y_slice_start, [num_samples, batch_size]) # [num_samples, batch_size]

    log_posterior = \
        tf.reduce_sum(y_dist.log_prob(y)) + \
        tf.reduce_sum(w_prior.log_prob(qw_masked))

    elbo = tf.reduce_sum(qw.distribution.entropy()) + log_posterior
    # elbo = log_posterior

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
    train = optimizer.minimize(-elbo)

    sess = tf.Session()

    niter = 1000
    feed_dict = dict()
    feed_dict[qw_loc_init] = qw_loc_init_value
    feed_dict[qw_scale_softminus_init] = qw_scale_softminus_init_value
    feed_dict[mask_init] = mask_init_value

    for batch_num in range(math.ceil(n/batch_size)):
        start_j = batch_num * batch_size
        fillmask(mask_init_value, start_j, batch_size)
        feed_dict[y_slice_start_init] = np.array([0, start_j], dtype=np.int32)

        sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)

        for t in range(niter):
            _, elbo_val = sess.run([train, elbo])
            print((t, elbo_val))
            # print(sess.run(tf.reduce_sum(y_dist.log_prob(y))))
            # print(sess.run(tf.reduce_sum(w_prior.log_prob(qw_masked))))
            # print(sess.run([tf.reduce_min(qw_loc), tf.reduce_max(qw_loc)]))
            # print(sess.run([tf.reduce_min(qw_scale), tf.reduce_max(qw_scale)]))


        print("batch")
        print(start_j)
        # print(sess.run(tf.reduce_max(qw.distribution.quantile(0.05))))
        # print(sess.run(tf.reduce_min(qw.distribution.quantile(0.95))))

        # print(sess.run(qw.distribution.quantile(0.05)) > 0)
        # print((sess.run(qw.distribution.quantile(0.05)) > 0) | (sess.run(qw.distribution.quantile(0.95)) < 0))

        # print(sess.run(qw_loc)[0:10])
        # print(sess.run(qw_scale)[0:10])

        print("nonzeros per transcript")
        # print(np.sum((sess.run(qw.distribution.quantile(0.05)) > 0) | (sess.run(qw.distribution.quantile(0.95)) < 0)) / batch_size)
        print(np.sum((sess.run(qw.distribution.quantile(0.01)) > 0)) / batch_size)
        print(np.sum((sess.run(qw.distribution.quantile(0.99)) < 0)) / batch_size)

        print(np.sum((sess.run(qw.distribution.quantile(0.001)) > 0)) / batch_size)
        print(np.sum((sess.run(qw.distribution.quantile(0.999)) < 0)) / batch_size)


        break

