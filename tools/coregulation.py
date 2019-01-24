
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
        scale=1e-3,
        # scale=qx_scale,
        # scale=100.0,
        name="qx")

    # variational estimate of w
    # ------------------
    qw_loc_init = tf.placeholder(tf.float32, (batch_size, n), name="qw_loc_init")
    qw_loc_init_value = np.zeros((batch_size, n), dtype=np.float32)
    qw_loc = tf.check_numerics(tf.Variable(qw_loc_init, name="qw_loc"), "qw_loc")
    # qw_loc = tf.Print(qw_loc, [tf.reduce_min(qw_loc), tf.reduce_max(qw_loc)], "qw_loc span")

    qw_scale_softminus_init = tf.placeholder(tf.float32, (batch_size, n), name="qw_scale_softminus_init")
    qw_scale_softminus_init_value = np.full((batch_size, n), -8.0, dtype=np.float32)
    qw_scale_param = tf.check_numerics(tf.nn.softplus(tf.Variable(qw_scale_softminus_init, name="qw_scale_param")), "qw_scale_param")
    # qw_scale_param = tf.Print(qw_scale_param, [tf.reduce_min(qw_scale_param), tf.reduce_max(qw_scale_param)], "qw_scale_param span")

    # [n, batch_size]
    qw = ed.Normal(
        loc=qw_loc,
        scale=qw_scale_param,
        name="qw")


    # variational estimate of w_scale
    # -----------------------
    qw_scale_loc_init_value = np.full((batch_size, n), -2.0, dtype=np.float32)
    qw_scale_loc_init = tf.placeholder(tf.float32, (batch_size, n), name="qw_scale_loc_init")
    qw_scale_loc = tf.Variable(qw_scale_loc_init, name="qw_scale_loc")
    # qw_scale_loc = tf.Print(qw_scale_loc, [tf.reduce_min(qw_scale_loc), tf.reduce_max(qw_scale_loc)], "qw_scale_loc")
    qw_scale_loc = tf.check_numerics(qw_scale_loc, "qw_scale_loc")
    # qw_scale_loc = tf.clip_by_value(qw_scale_loc, -10.0, 1000.0)

    qw_scale_scale_init_value = np.full((batch_size, n), -2.0, dtype=np.float32)
    qw_scale_scale_init = tf.placeholder(tf.float32, (batch_size, n), name="qw_scale_scale_init")
    qw_scale_scale = tf.check_numerics(tf.nn.softplus(tf.Variable(qw_scale_loc_init, name="qw_scale_scale")), "qw_scale_scale")

    qw_scale = ed.TransformedDistribution(
        distribution=
            tfp.distributions.Normal(
                loc=qw_scale_loc,
                scale=qw_scale_scale),
        bijector=tfp.bijectors.Exp(),
        name="qw_scale")
    # qw_scale = tf.Print(qw_scale, [tf.reduce_min(qw_scale), tf.reduce_max(qw_scale)], "qw_scale span")

    # variational estimate of b
    # ------------------

    qby_loc_init_value = np.empty([batch_size], dtype=np.float32)
    qby_loc_init = tf.placeholder(tf.float32, (batch_size,), name="qby_loc_init")
    qby_loc = tf.Variable(qby_loc_init, name="qby_loc")

    qby_scale_init_value = np.full([batch_size], -4.0, dtype=np.float32)
    qby_scale_init = tf.placeholder(tf.float32, (batch_size,), name="qby_scale_init")
    qby_scale = tf.nn.softplus(tf.Variable(qby_scale_init, name="qby_scale"))

    qby = ed.Normal(loc=qby_loc, scale=qby_scale, name="qby")


    qb_loc_init_value = np.empty([batch_size, 1, n], dtype=np.float32)
    qb_loc_init = tf.placeholder(tf.float32, (batch_size, 1, n), name="qb_loc_init")
    qb_loc = tf.Variable(qb_loc_init, name="qb_loc")

    qb_scale_init_value = np.full([batch_size, 1, n], -4.0, dtype=np.float32)
    qb_scale_init = tf.placeholder(tf.float32, (batch_size, 1, n), name="qb_scale_init")
    qb_scale = tf.nn.softplus(tf.Variable(qb_scale_init, name="qb_scale"))

    qb = ed.Normal(loc=qb_loc, scale=qb_scale, name="qb")
    # qb = ed.Normal(loc=qb_loc, scale=1e-5, name="qb")

    # w
    # -
    # w_scale_prior = tfd.Horseshoe(
    #     scale=1.0,
    #     name="w_scale_prior")
    w_scale_prior = tfd.HalfCauchy(
        loc=0.0,
        scale=1.0,
        name="w_scale_prior")

    w_prior = tfd.Normal(
        loc=0.0,
        scale=qw_scale + 1e-5)
        # scale=tf.clip_by_value(qw_scale, 1e-3, 1000.0))

    # [n, batch_size]
    mask_init = tf.placeholder(tf.float32, (batch_size, n), name="mask_init")
    mask_init_value = np.empty([batch_size, n], dtype=np.float32)
    mask = tf.Variable(mask_init, name="mask", trainable=False)

    # x_mean, x_var = tf.nn.moments(qx, axes=0) # [n]
    # x_var = tf.Print(x_var, [tf.reduce_min(x_var), tf.reduce_mean(x_var), tf.reduce_max(x_var)], "x_var")

    # x_std = (qx - x_mean) / tf.sqrt(x_var) # [num_samples, n]
    # x_std = qx - x_mean
    # x_std = tf.Print(x_std, [tf.reduce_min(x_std), tf.reduce_mean(x_std), tf.reduce_max(x_std)], "x_std")

    # mask = tf.Print(mask, [tf.reduce_min(mask), tf.reduce_max(mask)], "mask span")
    # qw_ = tf.Print(qw, [tf.reduce_min(qw), tf.reduce_max(qw)], "qw span")
    qw_ = qw
    qw_masked = qw_ * mask
    # qw_masked = tf.Print(qw_masked, [tf.reduce_min(qw_masked), tf.reduce_max(qw_masked)], "qw_masked span")

    # TRY: see if it at least finds a self edge for transcripts that vary
    # TODO:
    # qw_masked = qw

    # [num_samplrs, batch_size]
    err_scale = 0.25 # TODO: consider what to do with this shit

    # qb_ = tf.Print(qb, [tf.reduce_min(qb), tf.reduce_max(qb)], "qb span")
    # qby_ = tf.Print(qby, [tf.reduce_min(qby), tf.reduce_max(qby)], "qby span")

    qx_minus_qb = qx - qb
    # qx_minus_qb = tf.Print(qx_minus_qb, [tf.reduce_min(qx_minus_qb), tf.reduce_max(qx_minus_qb)], "qx_minus_qb span")
    qxqw = tf.matmul(qx_minus_qb, tf.expand_dims(qw_masked, axis=-1))
    # qxqw = tf.Print(qxqw, [tf.reduce_min(qxqw), tf.reduce_max(qxqw)], "qxqw span")


    y_dist_loc = qby + qxqw
    y_dist = tfd.Normal(
        loc=y_dist_loc,
        scale=err_scale)

    # TODO:
    y_slice_start_init = tf.placeholder(tf.int32, 2, name="y_slice_start_init") # set to [0, j]
    y_slice_start = tf.Variable(y_slice_start_init, name="y_slice_start", trainable=False)
    y = tf.slice(qx, y_slice_start, [num_samples, batch_size]) # [num_samples, batch_size]

    y_log_prob = tf.reduce_sum(y_dist.log_prob(y))

    w_log_prob = w_prior.log_prob(qw_masked)
    # w_log_prob = tf.Print(w_log_prob, [tf.reduce_min(w_log_prob), tf.reduce_max(w_log_prob)], "w_log_prob span")
    w_log_prob = tf.reduce_sum(w_log_prob)


    # w_log_prob = tf.reduce_sum(w_prior.log_prob(qw_masked))

    w_scale_log_prob = tf.reduce_sum(w_scale_prior.log_prob(qw_scale))

    # y_log_prob = tf.Print(y_log_prob, [y_log_prob], "y_log_prob")
    # w_log_prob = tf.Print(w_log_prob, [w_log_prob], "w_log_prob")
    # w_scale_log_prob = tf.Print(w_scale_log_prob, [w_scale_log_prob], "w_scale_log_prob")

    log_posterior = y_log_prob + w_log_prob + w_scale_log_prob

    entropy = \
        tf.reduce_sum(qb.distribution.entropy()) + \
        tf.reduce_sum(qby.distribution.entropy()) + \
        tf.reduce_sum(tf.log(qw_scale_scale * tf.exp(qw_scale_loc + 0.5)))
        # tf.reduce_sum(qw_scale.distribution.entropy()) # + \
        # tf.reduce_sum(qw.distribution.entropy())

    elbo = entropy + log_posterior
    # elbo = log_posterior

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    train = optimizer.minimize(-elbo)

    sess = tf.Session()

    niter = 10000
    feed_dict = dict()
    feed_dict[qw_scale_loc_init] = qw_scale_loc_init_value
    feed_dict[qw_scale_scale_init] = qw_scale_scale_init_value
    feed_dict[qw_loc_init] = qw_loc_init_value
    feed_dict[qw_scale_softminus_init] = qw_scale_softminus_init_value
    feed_dict[mask_init] = mask_init_value
    feed_dict[qb_loc_init] = qb_loc_init_value
    feed_dict[qb_scale_init] = qb_scale_init_value
    feed_dict[qby_loc_init] = qby_loc_init_value
    feed_dict[qby_scale_init] = qby_scale_init_value

    # check = tf.add_check_numerics_ops()

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
            # _, elbo_val = sess.run([train, elbo])
            _, elbo_val = sess.run([train, elbo], feed_dict=feed_dict)

            if t % 100 == 0:
                print((t, elbo_val))

                qw_scale_values = sess.run(qw_scale)
                print([np.min(qw_scale_values), np.max(qw_scale_values)])


                print(sess.run(y_log_prob))
                print(sess.run(w_log_prob))
                print(sess.run(w_scale_log_prob))

                qw_values, qw_loc_values, qw_scale_values = sess.run(
                    [tf.reshape(qw, [-1]), tf.reshape(qw_loc, [-1]), tf.reshape(qw_scale_param, [-1])])

                qw_values = qw_values[1:]
                qw_loc_values = qw_loc_values[1:]
                qw_scale_values = qw_scale_values[1:]

                idx = np.argmax(qw_scale_values)
                print(idx, qw_loc_values[idx], qw_scale_values[idx])

                print([np.min(qw_values), np.max(qw_values)])
                print([np.min(qw_loc_values), np.max(qw_loc_values)])
                print([np.min(qw_scale_values), np.max(qw_scale_values)])

                # print([np.max(qw_values), np.max(qw_loc_values), np.max(qw_scale_values)])

                # print(qw_values[0:5])
                # print(qw_loc_values[0:5])
                # print(qw_scale_values[0:5])

            # print(sess.run(qtau))
            # print(sess.run(qsigma))
            # print(sess.run(tf.reduce_sum(y_dist.log_prob(y))))
            # print(sess.run(tf.reduce_sum(w_prior.log_prob(qw_masked))))
            # print(sess.run([tf.reduce_min(qw_loc), tf.reduce_max(qw_loc)]))
            # print(sess.run([tf.reduce_min(qw_scale), tf.reduce_max(qw_scale)]))

            # print("----------------")
            # print(sess.run(log_posterior))
            # print(sess.run(entropy))

            # print(sess.run(y_log_prob))
            # print(sess.run(w_log_prob))
            # print(sess.run(w_scale_log_prob))

            # print((sess.run(tf.reduce_min(qw_scale_loc)), sess.run(tf.reduce_max(qw_scale_loc))))
            # print((sess.run(tf.reduce_min(qw_scale_scale)), sess.run(tf.reduce_max(qw_scale_scale))))

            # print((sess.run(tf.reduce_min(qw_scale)), sess.run(tf.reduce_max(qw_scale))))

            # qw_loc_values = sess.run(qw_loc)
            # qw_scale_values = sess.run(qw_scale_param)
            # y_dist_loc_values = sess.run(y_dist_loc)
            # print((np.min(qw_loc_values), np.max(qw_loc_values)))
            # print((np.min(qw_scale_values), np.max(qw_scale_values)))
            # print((np.min(y_dist_loc_values), np.max(y_dist_loc_values)))


        print("")
        print("batch")
        print(start_j)
        # print(sess.run(tf.reduce_max(qw.distribution.quantile(0.05))))
        # print(sess.run(tf.reduce_min(qw.distribution.quantile(0.95))))

        # print(sess.run(qw.distribution.quantile(0.05)) > 0)
        # print((sess.run(qw.distribution.quantile(0.05)) > 0) | (sess.run(qw.distribution.quantile(0.95)) < 0))

        # print(sess.run(qw_loc)[0:10])
        # print(sess.run(qw_scale)[0:10])

        # qw_loc_values = sess.run(qw_loc)
        # qw_scale_values = sess.run(qw_scale_param)
        # print((np.min(qw_loc_values), np.max(qw_loc_values)))
        # print((np.min(qw_scale_values), np.max(qw_scale_values)))

        lower_credible = sess.run(qw.distribution.quantile(0.001))[0,:]
        upper_credible = sess.run(qw.distribution.quantile(0.999))[0,:]

        # print("diagonal")
        # print(lower_credible[start_j])
        # print(upper_credible[start_j])

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
        if count > 0:
            break

        # break
        # if start_j >= 100:
        #     break

