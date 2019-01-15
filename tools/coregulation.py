
import numpy as np
import scipy
import scipy.stats as sps
import tensorflow as tf
import tensorflow_probability as tfp
import sys
import math
from tensorflow_probability import distributions as tfd
from tensorflow_probability import edward2 as ed

def fillmask(mask_init_value, start_j, batch_size):
    for (k, j) in enumerate(range(start_j, start_j+batch_size)):
        mask_init_value[k, :] = 0
        mask_init_value[k, j+1:] = 1


def estimate_gmm_precision(qx_loc, qx_scale, batch_size=1):
    num_samples = qx_loc.shape[0]
    n = qx_loc.shape[1]

    # [num_samples, n]
    qx = ed.Normal(
        loc=qx_loc,
        # scale=qx_scale,
        scale=100.0,
        name="qx")

    qw_loc_init = tf.placeholder(tf.float32, (batch_size, n), name="qw_loc_init")
    qw_loc_init_value = np.zeros((batch_size, n), dtype=np.float32)
    qw_loc = tf.Variable(qw_loc_init, name="qw_loc")

    qw_scale_softminus_init = tf.placeholder(tf.float32, (batch_size, n), name="qw_scale_softminus_init")
    qw_scale_softminus_init_value = np.full((batch_size, n), -8.0, dtype=np.float32)
    qw_scale = tf.nn.softplus(tf.Variable(qw_scale_softminus_init, name="qw_scale"))

    # [n, batch_size]
    qw = ed.Normal(
        loc=qw_loc,
        scale=qw_scale,
        name="qw")

    # sigma
    # -----
    qsigma_loc_init_value = np.full((1,), -1.0, dtype=np.float32)
    qsigma_loc_init = tf.placeholder(tf.float32, (1,), name="qsigma_loc_init")
    qsigma_loc = tf.Variable(qsigma_loc_init, name="qsigma_loc")

    qsigma_scale_init_value = np.full((1,), -1.0, dtype=np.float32)
    qsigma_scale_init = tf.placeholder(tf.float32, (1,), name="qsigma_scale_init")
    qsigma_scale = tf.Variable(qsigma_scale_init, name="qsigma_scale")

    qsigma = ed.TransformedDistribution(
        distribution=
            tfp.distributions.Normal(
                loc=qsigma_loc,
                scale=qsigma_scale),
        bijector=tfp.bijectors.Exp(),
        name="qsigma")

    # tau
    # ---
    tau_prior = tfd.HalfCauchy(
        loc=0.0,
        scale=qsigma)

    qtau_loc_init_value = np.full((1,), -1.0, dtype=np.float32)
    qtau_loc_init = tf.placeholder(tf.float32, (1,), name="qtau_loc_init")
    qtau_loc = tf.Variable(qtau_loc_init, name="qtau_loc")

    qtau_scale_init_value = np.full((1,), -1.0, dtype=np.float32)
    qtau_scale_init = tf.placeholder(tf.float32, (1,), name="qtau_scale_init")
    qtau_scale = tf.Variable(qtau_scale_init, name="qtau_scale")

    qtau = ed.TransformedDistribution(
        distribution=
            tfp.distributions.Normal(
                loc=qtau_loc,
                scale=qtau_scale),
        bijector=tfp.bijectors.Exp(),
        name="qtau")

    # w_scale
    # -------
    w_scale_prior = tfd.HalfCauchy(
        loc=tf.fill([batch_size, n], 0.0),
        scale=qtau)

    qw_scale_loc_init_value = np.full((batch_size, n), -8.0, dtype=np.float32)
    qw_scale_loc_init = tf.placeholder(tf.float32, (batch_size, n), name="qw_scale_loc_init")
    qw_scale_loc = tf.Variable(qw_scale_loc_init, name="qw_scale_loc")

    qw_scale_scale_init_value = np.full((batch_size, n), -2.0, dtype=np.float32)
    qw_scale_scale_init = tf.placeholder(tf.float32, (batch_size, n), name="qw_scale_scale_init")
    qw_scale_scale = tf.nn.softplus(tf.Variable(qw_scale_loc_init, name="qw_scale_scale"))

    qw_scale = ed.TransformedDistribution(
        distribution=
            tfp.distributions.Normal(
                loc=qw_scale_loc,
                scale=qw_scale_scale),
        bijector=tfp.bijectors.Exp(),
        name="qw_scale")

    # b
    # -

    qby_loc_init_value = np.empty([batch_size], dtype=np.float32)
    qby_loc_init = tf.placeholder(tf.float32, (batch_size,), name="qby_loc_init")
    qby_loc = tf.Variable(qby_loc_init, name="qby_loc")

    qby_scale_init_value = np.full([batch_size], -1.0, dtype=np.float32)
    qby_scale_init = tf.placeholder(tf.float32, (batch_size,), name="qby_scale_init")
    qby_scale = tf.nn.softplus(tf.Variable(qby_scale_init, name="qby_scale"))

    qby = ed.Normal(loc=qby_loc, scale=qby_scale, name="qby")


    qb_loc_init_value = np.empty([batch_size, 1, n], dtype=np.float32)
    qb_loc_init = tf.placeholder(tf.float32, (batch_size, 1, n), name="qb_loc_init")
    qb_loc = tf.Variable(qb_loc_init, name="qb_loc")

    qb_scale_init_value = np.full([batch_size, 1, n], -1.0, dtype=np.float32)
    qb_scale_init = tf.placeholder(tf.float32, (batch_size, 1, n), name="qb_scale_init")
    qb_scale = tf.nn.softplus(tf.Variable(qb_scale_init, name="qb_scale"))

    qb = ed.Normal(loc=qb_loc, scale=qb_scale, name="qb")


    # w
    # -
    w_prior = tfd.Normal(
        loc=0.0,
        scale=qw_scale)

    # [n, batch_size]
    mask_init = tf.placeholder(tf.float32, (batch_size, n), name="mask_init")
    mask_init_value = np.empty([batch_size, n], dtype=np.float32)
    mask = tf.Variable(mask_init, name="mask", trainable=False)

    x_mean, x_var = tf.nn.moments(qx, axes=0) # [n]
    # x_var = tf.Print(x_var, [tf.reduce_min(x_var), tf.reduce_mean(x_var), tf.reduce_max(x_var)], "x_var")

    # x_std = (qx - x_mean) / tf.sqrt(x_var) # [num_samples, n]
    # x_std = qx - x_mean
    # x_std = tf.Print(x_std, [tf.reduce_min(x_std), tf.reduce_mean(x_std), tf.reduce_max(x_std)], "x_std")

    qw_masked = qw * mask
    # TRY: see if it at least finds a self edge for transcripts that vary
    # qw_masked = qw

    # [num_samplrs, batch_size]
    err_scale = 0.5 # TODO: consider what to do with this shit
    y_dist = tfd.Normal(
        loc=qby + tf.matmul(qx - qb, tf.expand_dims(qw_masked, axis=-1)),
        scale=err_scale)

    # TODO:
    y_slice_start_init = tf.placeholder(tf.int32, 2, name="y_slice_start_init") # set to [0, j]
    y_slice_start = tf.Variable(y_slice_start_init, name="y_slice_start", trainable=False)
    y = tf.slice(qx, y_slice_start, [num_samples, batch_size]) # [num_samples, batch_size]

    log_posterior = \
        tf.reduce_sum(y_dist.log_prob(y)) + \
        tf.reduce_sum(w_prior.log_prob(qw_masked)) + \
        tf.reduce_sum(w_scale_prior.log_prob(qw_scale)) + \
        tf.reduce_sum(tau_prior.log_prob(qtau)) + \
        tf.reduce_sum(-2 * tf.log(qsigma)) # jefferys prior

    elbo = tf.reduce_sum(qw.distribution.entropy()) + log_posterior
    # elbo = log_posterior

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
    train = optimizer.minimize(-elbo)

    sess = tf.Session()

    niter = 5000
    feed_dict = dict()
    feed_dict[qsigma_loc_init] = qsigma_loc_init_value
    feed_dict[qsigma_scale_init] = qsigma_scale_init_value
    feed_dict[qtau_loc_init] = qtau_loc_init_value
    feed_dict[qtau_scale_init] = qtau_scale_init_value
    feed_dict[qw_scale_loc_init] = qw_scale_loc_init_value
    feed_dict[qw_scale_scale_init] = qw_scale_scale_init_value
    feed_dict[qw_loc_init] = qw_loc_init_value
    feed_dict[qw_scale_softminus_init] = qw_scale_softminus_init_value
    feed_dict[mask_init] = mask_init_value
    feed_dict[qb_loc_init] = qb_loc_init_value
    feed_dict[qb_scale_init] = qb_scale_init_value
    feed_dict[qby_loc_init] = qby_loc_init_value
    feed_dict[qby_scale_init] = qby_scale_init_value

    qx_loc_means = np.mean(qx_loc, axis=0)

    count = 0
    for batch_num in range(math.ceil(n/batch_size)):
        start_j = batch_num * batch_size

        fillmask(mask_init_value, start_j, batch_size)
        feed_dict[y_slice_start_init] = np.array([0, start_j], dtype=np.int32)

        for k in range(batch_size):
            qb_loc_init_value[k,0,:] = qx_loc_means
            qby_loc_init_value[k] = qx_loc_means[start_j+k]

        sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)

        for t in range(niter):
            _, elbo_val = sess.run([train, elbo])
            # print((t, elbo_val))
            # print(sess.run(qtau))
            # print(sess.run(qsigma))
            # print(sess.run(tf.reduce_sum(y_dist.log_prob(y))))
            # print(sess.run(tf.reduce_sum(w_prior.log_prob(qw_masked))))
            # print(sess.run([tf.reduce_min(qw_loc), tf.reduce_max(qw_loc)]))
            # print(sess.run([tf.reduce_min(qw_scale), tf.reduce_max(qw_scale)]))

        print("")
        print("batch")
        print(start_j)
        # print(sess.run(tf.reduce_max(qw.distribution.quantile(0.05))))
        # print(sess.run(tf.reduce_min(qw.distribution.quantile(0.95))))

        # print(sess.run(qw.distribution.quantile(0.05)) > 0)
        # print((sess.run(qw.distribution.quantile(0.05)) > 0) | (sess.run(qw.distribution.quantile(0.95)) < 0))

        # print(sess.run(qw_loc)[0:10])
        # print(sess.run(qw_scale)[0:10])

        lower_credible = sess.run(qw.distribution.quantile(0.001))[0,:]
        upper_credible = sess.run(qw.distribution.quantile(0.999))[0,:]

        print("diagonal")
        print(lower_credible[start_j])
        print(upper_credible[start_j])

        print("credible span")
        print(np.max(lower_credible))
        print(np.min(upper_credible))

        print("nonzeros per transcript")
        # print(np.sum((sess.run(qw.distribution.quantile(0.05)) > 0) | (sess.run(qw.distribution.quantile(0.95)) < 0)) / batch_size)
        # print(np.sum((sess.run(qw.distribution.quantile(0.01)) > 0)) / batch_size)
        # print(np.sum((sess.run(qw.distribution.quantile(0.99)) < 0)) / batch_size)

        print(np.sum((lower_credible > 0.1)) / batch_size)
        pos_idxs = np.array(range(n))[lower_credible > 0.1]
        # print(pos_idxs)

        print(np.sum((upper_credible < -0.1)) / batch_size)
        neg_idxs = np.array(range(n))[upper_credible < -0.1]
        # print(neg_idxs)

        # print([(l,u) for (l,u) in zip(lower_credible[pos_idxs], upper_credible[pos_idxs])])
        # print([(l,u) for (l,u) in zip(lower_credible[neg_idxs], upper_credible[neg_idxs])])



        # us = qx_loc[:,start_j]
        # vs = qx_loc[:,pos_idxs[0]]
        # ws = qx_loc[:,neg_idxs[0]]

        # print((lower_credible[pos_idxs[0]], upper_credible[pos_idxs[0]]))
        # print((lower_credible[neg_idxs[0]], upper_credible[neg_idxs[0]]))

        # print("qby")
        # print(sess.run(qby_loc)[0])

        # print("qb")
        # print(sess.run(qb_loc)[0, 0, pos_idxs[0]])
        # print(sess.run(qb_loc)[0, 0, neg_idxs[0]])


        # print(us)
        # print(sps.pearsonr(us, vs))
        # print(vs)
        # print(sps.pearsonr(us, ws))
        # print(ws)




        # us = (us - np.mean(us)) / np.sqrt(np.var(us))
        # vs = (vs - np.mean(vs)) / np.sqrt(np.var(vs))
        # ws = (ws - np.mean(ws)) / np.sqrt(np.var(ws))

        # us = (us - np.mean(us))
        # vs = (vs - np.mean(vs))
        # ws = (ws - np.mean(ws))

        # print(us)
        # print(sps.pearsonr(us, vs))
        # print(vs)
        # print(sps.pearsonr(us, ws))
        # print(ws)

        count += 1
        if count > 20:
            break

        # break
        # if start_j >= 100:
        #     break

