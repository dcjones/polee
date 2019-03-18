
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
        # anything but self-edges
        mask_init_value[k, :] = 1
        mask_init_value[k, j] = 0

        # alternative: only indexes > j
        # mask_init_value[k, :] = 0
        # mask_init_value[k, j+1:] = 1

        # debugging: allow soft-edges
        # mask_init_value[k, j:] = 1

        # debugging: only allow self-edges
        # mask_init_value[k, j] = 1


def estimate_gmm_precision(
        qx_loc, qx_scale, fixed_expression=False,
        profile_trace=False, tensorboard_summaries=False,
        batch_size=100, err_scale=0.2, edge_cutoff=0.7):
    num_samples = qx_loc.shape[0]
    n = qx_loc.shape[1]

    batch_size = min(batch_size, n)

    # [num_samples, n]
    if fixed_expression:
        qx = qx_loc
    else:
        qx = ed.Normal(
            loc=qx_loc,
            scale=qx_scale,
            name="qx")

    b = np.mean(qx_loc, axis=0)

    # variational estimate of w
    # -------------------------
    qw_loc_init = tf.placeholder(tf.float32, (batch_size, n), name="qw_loc_init")
    qw_loc_init_value = np.zeros((batch_size, n), dtype=np.float32)
    qw_loc = tf.Variable(qw_loc_init, name="qw_loc")
    qw = qw_loc

    # variational estimate of w_scale
    # -------------------------------

    qw_scale_loc_init_value = np.full((batch_size, n), -3.0, dtype=np.float32)
    qw_scale_loc_init = tf.placeholder(tf.float32, (batch_size, n), name="qw_scale_loc_init")
    qw_scale_loc = tf.Variable(qw_scale_loc_init, name="qw_scale_loc")
    qw_scale = tf.nn.softplus(qw_scale_loc)

    # estimate of b
    # -------------

    by_init_value = np.zeros((batch_size,), dtype=np.float32)
    by_init = tf.placeholder(tf.float32, (batch_size,), name="by_init")
    by = tf.Variable(by_init, name="by", trainable=False) # [batch_size]

    # w
    # -

    w_scale_prior = tfd.HalfCauchy(
        loc=0.0,
        scale=1.0,
        name="w_scale_prior")

    # qw_scale can be shrunk all the way to zero, producing NaNs
    qw_scale = tf.clip_by_value(qw_scale, 1e-4, 10000.0)

    scale_tau = 0.1
    w_prior = tfd.Normal(
        loc=0.0,
        scale=qw_scale * scale_tau,
        name="w_prior")


    # [n, batch_size]
    mask_init = tf.placeholder(tf.float32, (batch_size, n), name="mask_init")
    mask_init_value = np.empty([batch_size, n], dtype=np.float32)
    mask = tf.Variable(mask_init, name="mask", trainable=False)

    qw_masked = qw * mask # [batch_size, n]

    qx_std = qx - b # [num_samples, n]

    # CONDITIONAL CORRELATION
    # qxqw = tf.matmul(qx_std, qw_masked, transpose_b=True) # [num_samples, batch_size]
    # y_dist_loc = qxqw + by

    # UNCONDITIONAL CORRELATION
    qxqw = tf.expand_dims(qx_std, 1) * tf.expand_dims(qw_masked, 0) # [num_samples, num_batches, n]
    y_dist_loc = tf.expand_dims(tf.expand_dims(by, 0), -1) + qxqw # [num_samples, num_batches, n]

    y_dist = tfd.StudentT(
        loc=y_dist_loc,
        scale=err_scale,
        df=10.0)

    y_slice_start_init = tf.placeholder(tf.int32, 2, name="y_slice_start_init") # set to [0, j]
    y_slice_start = tf.Variable(y_slice_start_init, name="y_slice_start", trainable=False)
    y = tf.slice(qx, y_slice_start, [num_samples, batch_size]) # [num_samples, batch_size]

    # y = tf.Print(y, [tf.square(y_dist_loc - tf.expand_dims(y, -1))], "y", summarize=16)

    # objective function
    # ------------------

    y = tf.expand_dims(y, -1)
    y_log_prob       = tf.reduce_sum(y_dist.log_prob(y))
    w_log_prob       = tf.reduce_sum(w_prior.log_prob(qw_masked))
    w_scale_log_prob = tf.reduce_sum(w_scale_prior.log_prob(qw_scale))
    log_posterior = y_log_prob + w_log_prob + w_scale_log_prob
    elbo = log_posterior

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
    train = optimizer.minimize(-elbo)

    sess = tf.Session()

    niter = 1000
    feed_dict = dict()
    feed_dict[qw_scale_loc_init] = qw_scale_loc_init_value
    feed_dict[qw_loc_init] = qw_loc_init_value
    feed_dict[mask_init] = mask_init_value
    feed_dict[by_init] = by_init_value

    qx_loc_means = np.mean(qx_loc, axis=0)

    # check_ops = tf.add_check_numerics_ops()
    if tensorboard_summaries:
        # tf.summary.histogram("qw_loc_param", qw_loc)
        # tf.summary.histogram("qw_scale_param", qw_scale_param)
        tf.summary.scalar("y_log_prob", y_log_prob)
        tf.summary.scalar("w_log_prob", w_log_prob)
        tf.summary.scalar("w_scale_log_prob", w_scale_log_prob)

        tf.summary.scalar("qw min", tf.reduce_min(qw))
        tf.summary.scalar("qw max", tf.reduce_max(qw))
        tf.summary.scalar("qw_scale min", tf.reduce_min(qw_scale))
        tf.summary.scalar("qw_scale max", tf.reduce_max(qw_scale))

        # tf.summary.histogram("qw_scale_loc_param", qw_scale_loc)
        # tf.summary.histogram("qw_scale_scale_param", qw_scale_scale)

    edges = dict()

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
            by_init_value[k] = b[start_j+k]

        sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)

        # if requested, just benchmark one run of the training operation and return
        if profile_trace:
            print("WRITING PROFILING DATA")
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            sess.run(train, options=options, run_metadata=run_metadata)
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open('log/timeline.json', 'w') as f:
                f.write(chrome_trace)
            break

        if tensorboard_summaries:
            train_writer = tf.summary.FileWriter("log/" + "batch-" + str(batch_num), sess.graph)
            tf.summary.scalar("elbo", elbo)
            merged_summary = tf.summary.merge_all()

        for t in range(niter):
            # _, elbo_val = sess.run([train, elbo])
            # _, entropy_val, log_posterior_val, elbo_val = sess.run([train, entropy, log_posterior, elbo])
            _, y_log_prob_value, w_log_prob_value, w_scale_log_prob_value = sess.run([train, y_log_prob, w_log_prob, w_scale_log_prob])
            if t % 100 == 0:
                # print((t, elbo_val, log_posterior_val, entropy_val))
                print((y_log_prob_value, w_log_prob_value, w_scale_log_prob_value))
                # print((t, elbo_val))
            if tensorboard_summaries:
                train_writer.add_summary(sess.run(merged_summary), t)

        print("")
        print("batch")
        print(start_j)

        # qw_scale_min, qw_scale_mean, qw_scale_max = sess.run(
        #     [tf.reduce_min(qw_scale), tf.reduce_mean(qw_scale), tf.reduce_max(qw_scale)])
        # print(("qw_scale span", qw_scale_min, qw_scale_mean, qw_scale_max))

        # lower_credible = sess.run(qw.distribution.quantile(0.01))
        # upper_credible = sess.run(qw.distribution.quantile(0.99))
        lower_credible = upper_credible = sess.run(qw)

        print("credible span")
        print(np.max(lower_credible))
        print(np.min(upper_credible))

        print("nonzeros")
        print(np.sum((lower_credible > edge_cutoff)))
        print(np.sum((upper_credible < -edge_cutoff)))

        for k in range(batch_size):
            neighbors = []
            for j in range(n):
                if lower_credible[k, j] > edge_cutoff or upper_credible[k, j] < -edge_cutoff:
                    neighbors.append((j, lower_credible[k, j], upper_credible[k, j]))
            edges[start_j+k] = neighbors

        count += 1
        if count > 4:
            break

    return edges


