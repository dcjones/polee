
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sys
from tensorflow_probability import distributions as tfd
from tensorflow_probability import edward2 as ed

def estimate_gmm_precision(qx_loc, qx_scale):
    num_samples = qx_loc.shape[0]
    n = qx_loc.shape[1]

    qx = ed.Normal(
        loc=qx_loc,
        scale=qx_scale,
        name="qx")

    qw_loc_init = tf.placeholder(tf.float32, (n, 1), name="qw_loc_init")
    qw_loc = tf.Variable(qw_loc_init, "qw_loc")

    qw_scale_softminus_init = tf.placeholder(tf.float32, (n, 1), name="qw_scale_softminus_init")
    qw_scale = tf.nn.softplus(tf.Variable(qw_scale_softminus_init, "qw_scale"))

    qw = ed.Normal(
        loc=qw_loc,
        scale=qw_scale,
        name="qw")

    # TODO: maybe try something like horseshoe prior
    w_prior = tfd.Normal(
        loc=0.0,
        scale=0.1)

    err_scale = 0.1

    # deselect and select matrices: the format is identity except for a zero at
    # (j,j), the latter is zeros except for a one at (j, j)
    #
    # The trick here is to set the column being predicted to ones, and the
    # corresponding entry in w then becomes the bias variable in the regression.

    D_init = tf.placeholder(tf.float32, (1, n), name="D_init")
    D = tf.Variable(D_init, name="D", trainable=False)

    S_init = tf.placeholder(tf.float32, (1, n), name="S_init")
    S = tf.Variable(S_init, name="S", trainable=False)

    x_mean, x_var = tf.nn.moments(qx, axes=0)
    x_std = (qx - x_mean) / x_var

    u = x_std*D

    y_dist = tfd.Normal(
        loc=tf.matmul(u, qw),
        scale=err_scale)

    y_slice_start_init = tf.placeholder(tf.int32, 2, name="y_slice_start_init") # set to [0, j]
    y_slice_start = tf.Variable(y_slice_start_init, name="y_slice_start", trainable=False)
    y = tf.slice(x_std, y_slice_start, [num_samples, 1])

    log_posterior = \
        tf.reduce_sum(y_dist.log_prob(y)) + \
        tf.reduce_sum(w_prior.log_prob(qw))

    elbo = tf.reduce_sum(qw.distribution.entropy()) - log_posterior
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
    train = optimizer.minimize(elbo)

    sess = tf.Session()

    niter = 100
    feed_dict = dict()
    for j in range(n):
        # TODO: for efficiency we should try doing multiple j's at once.

        # TODO: what we should do to get a well-defined DAG is to condition
        # x_j only on all x_k where k > j

        # feed_dict[qw_loc_init] = np.random.randn(n, 1)
        feed_dict[qw_loc_init] = np.zeros((n,1), dtype=np.float32)
        feed_dict[qw_scale_softminus_init] = np.full((n, 1), -2.0, dtype=np.float32)
        tmp = np.ones((1, n), dtype=np.float32)
        tmp[0,j] = 0.0
        feed_dict[D_init] = tmp
        tmp = np.zeros((1, n), dtype=np.float32)
        tmp[0,j] = 1.0
        feed_dict[S_init] = tmp
        feed_dict[y_slice_start_init] = np.array([0, j], dtype=np.int32)

        sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)

        for t in range(niter):
            _, elbo_val = sess.run([train, elbo])
            # print((t, elbo_val))

        print(j)
        print(sess.run(tf.reduce_max(qw.distribution.quantile(0.05))))
        print(sess.run(tf.reduce_min(qw.distribution.quantile(0.95))))

        # print(sess.run([tf.reduce_min(qw_loc), tf.reduce_max(qw_loc)]))
        # print(sess.run([tf.reduce_min(qw_scale), tf.reduce_max(qw_scale)]))

        if j > 100:
            break

