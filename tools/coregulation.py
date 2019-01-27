
import numpy as np
import scipy
import scipy.stats as sps
import tensorflow as tf
import tensorflow_probability as tfp
import sys
import math
from tensorflow_probability import distributions as tfd
from tensorflow_probability import edward2 as ed
from tensorflow.python.client import timeline


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


def fillmask(mask_init_value, start_j, batch_size):
    for (k, j) in enumerate(range(start_j, start_j+batch_size)):
        mask_init_value[k, :] = 0
        mask_init_value[k, j+1:] = 1


def estimate_gmm_precision(qx_loc, qx_scale, batch_size=5, err_scale=0.25):
    num_samples = qx_loc.shape[0]
    n = qx_loc.shape[1]

    batch_size = min(batch_size, n)

    # [num_samples, n]
    qx = ed.Normal(
        loc=qx_loc,
        scale=qx_scale,
        # scale=1e-4,
        name="qx")

    b = np.mean(qx_loc, axis=0)

    # variational estimate of w
    # -------------------------
    qw_loc_init = tf.placeholder(tf.float32, (batch_size, n), name="qw_loc_init")
    qw_loc_init_value = np.zeros((batch_size, n), dtype=np.float32)
    qw_loc = tf.Variable(qw_loc_init, name="qw_loc")

    qw_scale_softminus_init = tf.placeholder(tf.float32, (batch_size, n), name="qw_scale_softminus_init")
    qw_scale_softminus_init_value = np.full((batch_size, n), -4.0, dtype=np.float32)
    qw_scale_param = tf.nn.softplus(tf.Variable(qw_scale_softminus_init, name="qw_scale_param"))

    # [n, batch_size]
    qw = ed.Normal(
        loc=qw_loc,
        scale=qw_scale_param,
        name="qw")

    # variational estimate of w_scale
    # -------------------------------

    qw_scale_loc_init_value = np.full((batch_size, n), -6.0, dtype=np.float32)
    qw_scale_loc_init = tf.placeholder(tf.float32, (batch_size, n), name="qw_scale_loc_init")
    qw_scale_loc = tf.Variable(qw_scale_loc_init, name="qw_scale_loc")

    qw_scale_scale_init_value = np.full((batch_size, n), -4.0, dtype=np.float32)
    qw_scale_scale_init = tf.placeholder(tf.float32, (batch_size, n), name="qw_scale_scale_init")
    qw_scale_scale = tf.nn.softplus(tf.Variable(qw_scale_loc_init, name="qw_scale_scale"))

    qw_scale = ed.TransformedDistribution(
        distribution=
            tfp.distributions.Normal(
                loc=qw_scale_loc,
                scale=qw_scale_scale),
        bijector=tfp.bijectors.Exp(),
        name="qw_scale")


    # variational estimate of b
    # -------------------------

    by_init_value = np.zeros((batch_size,), dtype=np.float32)
    by_init = tf.placeholder(tf.float32, (batch_size,), name="by")
    by = tf.Variable(by_init, name="by")

    # w
    # -

    w_scale_prior = tfd.HalfCauchy(
        loc=0.0,
        scale=qw_scale,
        name="w_scale_prior")


    qw_scale_ = tf.Print(qw_scale,
        [tf.reduce_min(qw_scale), tf.reduce_max(qw_scale)], "qw_scale")

    w_prior = tfd.Normal(
        loc=0.0,
        # scale=qw_scale_,
        scale=0.1,
        name="w_prior")

    # [n, batch_size]
    mask_init = tf.placeholder(tf.float32, (batch_size, n), name="mask_init")
    mask_init_value = np.empty([batch_size, n], dtype=np.float32)
    mask = tf.Variable(mask_init, name="mask", trainable=False)

    # TODO:
    # qw_masked = qw * mask
    qw_masked = qw

    # [num_samples, batch_size]
    qx_minus_b = qx - b
    # qx_minus_b = tf.Print(qx_minus_b, [tf.reduce_min(qx_minus_b), tf.reduce_max(qx_minus_b)], "qx_minus_b")
    qxqw = tf.matmul(qx_minus_b, qw_masked, transpose_b=True)

    y_dist_loc = by + qxqw
    y_dist = tfd.Normal(
        loc=y_dist_loc,
        scale=err_scale)

    y_slice_start_init = tf.placeholder(tf.int32, 2, name="y_slice_start_init") # set to [0, j]
    y_slice_start = tf.Variable(y_slice_start_init, name="y_slice_start", trainable=False)
    y = tf.slice(qx, y_slice_start, [num_samples, batch_size]) # [num_samples, batch_size]

    # objective function
    # ------------------

    # log posterior
    # y = tf.Print(y, [tf.reduce_min(y_dist_loc - y), tf.reduce_max(y_dist_loc - y)], "y diff")
    y_log_prob = tf.reduce_sum(y_dist.log_prob(y))


    qw_ = tf.Print(qw,
        [tf.reduce_min(qw), tf.reduce_max(qw)], "qw")
    w_log_prob = tf.reduce_sum(w_prior.log_prob(qw_))

    # w_scale_log_prob = tf.reduce_sum(w_scale_prior.log_prob(tf.transpose(qw_scale)))
    w_scale_log_prob = tf.reduce_sum(w_scale_prior.log_prob(qw_scale))

    log_posterior = y_log_prob + w_log_prob + w_scale_log_prob

    # entropy
    qw_scale_entropy = tf.reduce_sum(tf.log(qw_scale_scale * tf.exp(qw_scale_loc + 0.5)))
    # mask this to avoid optimizing entropy on the unused elements

    # TODO:
    # qw_entropy = tf.reduce_sum(mask * qw.distribution.entropy())
    qw_entropy = tf.reduce_sum(qw.distribution.entropy())
    entropy = qw_entropy + qw_scale_entropy

    elbo = entropy + log_posterior

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    train = optimizer.minimize(-elbo)

    sess = tf.Session()

    niter = 5000
    feed_dict = dict()
    feed_dict[qw_scale_loc_init] = qw_scale_loc_init_value
    feed_dict[qw_scale_scale_init] = qw_scale_scale_init_value
    feed_dict[qw_loc_init] = qw_loc_init_value
    feed_dict[qw_scale_softminus_init] = qw_scale_softminus_init_value
    feed_dict[mask_init] = mask_init_value
    feed_dict[by_init] = by_init_value

    qx_loc_means = np.mean(qx_loc, axis=0)

    # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()
    # check_ops = tf.add_check_numerics_ops()

    count = 0
    num_batches = math.ceil(n/batch_size)
    for batch_num in range(num_batches):
        # deal with n not necessarily being divisible by batch_size
        if batch_num == num_batches - 1:
            start_j = n - batch_size
        else:
            start_j = batch_num * batch_size

        fillmask(mask_init_value, start_j, batch_size)
        feed_dict[y_slice_start_init] = np.array([0, start_j], dtype=np.int32)

        for k in range(batch_size):
            # qb_loc_init_value[k,0,:] = qx_loc_means
            # qby_loc_init_value[k] = qx_loc_means[start_j+k]
            by_init_value[k] = b[start_j+k]

        sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)


        # sess.run(elbo, options=options, run_metadata=run_metadata)
        # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        # chrome_trace = fetched_timeline.generate_chrome_trace_format()
        # with open('log/timeline.json', 'w') as f:
        #     f.write(chrome_trace)
        # sys.exit()

        for t in range(niter):
            _, elbo_val = sess.run([train, elbo])
            if t % 100 == 0:
                print((t, elbo_val))

        print("")
        print("batch")
        print(start_j)

        lower_credible = sess.run(qw.distribution.quantile(0.001))
        upper_credible = sess.run(qw.distribution.quantile(0.999))

        print("credible span")
        print(np.max(lower_credible))
        print(np.min(upper_credible))

        print("nonzeros")
        print(np.sum((lower_credible > 0.1)))
        print(np.sum((upper_credible < -0.1)))

        count += 1
        if count > 0:
            break


