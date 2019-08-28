
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from polee_approx_likelihood import *
from polee_training import *
import polee_regression
import sys



def classification_model(x, vars, n, K, num_samples, keep_prob):

    # lyr1 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
    # lyr1_output = lyr1(x)

    # drp1 = tf.keras.layers.Dropout(rate=dropout_rate)
    # drp1_output = drp1(lyr1_output)

    # lyr2 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
    # lyr2_output = lyr2(drp1_output)

    # drp2 = tf.keras.layers.Dropout(rate=dropout_rate)
    # drp2_output = drp2(lyr2_output)

    # lyrn = tf.keras.layers.Dense(K)
    # z_predict_logits = lyrn(drp2_output)

    # return z_predict_logits, lyr1, lyr2, lyrn

    lyr1 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
    lyr1_output = lyr1(x)

    drp1 = tf.keras.layers.Dropout(rate=1.0 - keep_prob)
    drp1_output = drp1(lyr1_output)

    lyr2 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
    lyr2_output = lyr2(x)

    drp2 = tf.keras.layers.Dropout(rate=1.0 - keep_prob)
    drp2_output = drp2(lyr2_output)
    # drp2_output = drp2(x)

    # lyr3 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
    # lyr3_output = lyr3(drp2_output)

    # drp3 = tf.keras.layers.Dropout(rate=dropout_rate)
    # drp3_output = drp2(lyr3_output)

    lyrn = tf.keras.layers.Dense(K)
        # kernel_regularizer=tf.keras.regularizers.l2(0.01))
    # z_predict_logits = lyrn(drp2_output)
    z_predict_logits = lyrn(x) # logistic regression

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
        sample_scales, use_posterior_mean):
    print("train_classifier")

    num_factors = z_true.shape[1] # number of categories


    if use_posterior_mean:
        x = tf.constant(x_init, name="x")
        # x = tf.exp(tf.constant(x_init, name="x"))
    else:
        # x = rnaseq_approx_likelihood_sampler_from_vars(num_samples, n, vars)

        # Seems to do much worse on log scale
        x = tf.log(rnaseq_approx_likelihood_sampler_from_vars(num_samples, n, vars))
        # x = rnaseq_approx_likelihood_sampler_from_vars(num_samples, n, vars)

    # x += tf.expand_dims(sample_scales, -1)



    # num_features = x_init.shape[1]
    # x_pm = tf.Variable(tf.zeros([num_samples, num_features]))
    # init = tf.global_variables_initializer()
    # sess.run(init, feed_dict=init_feed_dict)
    # assign_op = tf.assign(x_pm, x_pm + x)
    # niter = 200
    # for i in range(niter):
    #     sess.run(assign_op)
    # x_pm = x_pm / niter

    # # print(sess.run(tfp.stats.quantiles(x, 10, axis=1)))
    # print(sess.run(tfp.stats.quantiles(x_pm, 10, axis=1)))
    # x = tf.constant(sess.run(x_pm))



    # z_predict_logits, lyr1, lyr2, lyr3, lyrn = classification_model(x, vars, n, K, num_samples, 0.5)
    z_predict_logits, lyr1, lyr2, lyrn = classification_model(x, vars, n, num_factors, num_samples, 0.5)

    # z_predict_logits = tf.Print(
    #     z_predict_logits, [tf.reduce_min(z_predict_logits), tf.reduce_max(z_predict_logits)], "z_predict_logits span")

    loss = tf.losses.softmax_cross_entropy(z_true, logits=z_predict_logits)

    # train(sess, loss, init_feed_dict, 1000, 1e-3)
    # train(sess, loss, init_feed_dict, 10, 1e-3)
    train(sess, loss, init_feed_dict, 20000, 1e-1)

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
        sess, classify_model, init_feed_dict, num_samples, n, vars, x_init, num_factors,
        sample_scales, use_posterior_mean):

    if use_posterior_mean:
        x = tf.constant(x_init, name="x")
    else:
        # x = rnaseq_approx_likelihood_sampler_from_vars(num_samples, n, vars)
        x = tf.log(rnaseq_approx_likelihood_sampler_from_vars(num_samples, n, vars))

    # x += tf.expand_dims(sample_scales, -1)

    # z_predict_logits, lyr1, lyr2, lyr3, lyrn = classification_model(x, vars, n, K, num_samples, 0.0)
    z_predict_logits, lyr1, lyr2, lyrn = classification_model(x, vars, n, num_factors, num_samples, 1.0)

    # z_predict_logits = tf.Print(
    #     z_predict_logits, [tf.reduce_min(z_predict_logits), tf.reduce_max(z_predict_logits)], "z_predict_logits span")

    # Idea here is to run this a bunch of times and average the results
    z_mean = tf.Variable(tf.zeros([num_samples, num_factors]))

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

    accum_op = tf.assign(z_mean, z_mean + tf.nn.softmax(z_predict_logits, axis=1))
    niter = 100
    prog = Progbar(50, niter)
    for iter in range(niter):
        sess.run(accum_op)
        prog.update(iter, loss=0.0)
    z_mean /= niter

    return sess.run(z_mean)



# Alternative classifier idea:
# Set it up basically like the regression model.
#   1. Train with F fixed.
#   2. Then classify by making F variable 
#

scale_spline_degree = 10

def choose_spline_hinges(low, high):
    x_scale_hinges = []
    d = (high - low) / (scale_spline_degree+1)
    for i in range(scale_spline_degree):
        x_scale_hinges.append(low + (i+1)*d)
    return x_scale_hinges


# def train_probabalistic_classifier(
#         sess, init_feed_dict, num_samples, num_features, vars, x_init, Z,
#         sample_scales, use_point_estimates):

#     print("train_probabalistic_classifier")

def train_probabalistic_classifier(
        sess, init_feed_dict, num_samples, num_features, vars, x_init, Z,
        sample_scales, use_point_estimates):

    print("train_probabalistic_classifier")

    x_loc, w_loc, w_scale, x_bias_loc, x_scale_loc = \
        polee_regression.estimate_transcript_linear_regression(init_feed_dict, vars, x_init, Z,
        # sample_scales, use_point_estimates, sess, 2000)
        tf.expand_dims(sample_scales, -1), use_point_estimates, sess, 5000)

    params = {
        "x_bias": x_bias_loc,
        "w": w_loc,
        "x_scale": x_scale_loc }

    return params


# maximum posterior training
if False:
    def train_probabalistic_classifier(
            sess, init_feed_dict, num_samples, num_features, vars, x_init, Z,
            sample_scales, use_point_estimates):

        print("train_probabalistic_classifier")

        x_init_mean = np.mean(x_init, axis=0)
        x_scale_hinges = choose_spline_hinges(np.min(x_init_mean), np.max(x_init_mean))

        num_factors = Z.shape[1]

        w_global_scale = tf.maximum(1e-6, tf.nn.softplus(tf.Variable(0.0, name="w_global_scale")))
        w_global_scale_prior = tfd.HalfCauchy(loc=0.0, scale=1.0, name="w_global_scale_prior")

        w_local_scale = tf.maximum(1e-6, tf.nn.softplus(tf.Variable(tf.zeros(num_features), name="w_local_scale")))
        w_local_scale_prior = tfd.HalfCauchy(loc=0.0, scale=1.0, name="w_local_scale_prior")

        w = tf.Variable(tf.zeros([num_features, num_factors]), name="w")
        w_prior = tfd.Normal(
            loc=0.0,
            scale=tf.expand_dims(w_global_scale * w_local_scale, -1),
            # scale=tf.expand_dims(w_local_scale, -1),
            name="w_prior")
        # w_prior = tfd.StudentT(
            # df=1.0,
        # w_prior = tfd.Normal(
        #     loc=0.0,
        #     scale=0.2,
        #     name="w_prior")

        x_bias = tf.Variable(
            tf.reduce_mean(x_init, axis=0),
            name="x_bias")
        x_bias_prior = tfd.Normal(
            loc=np.float32(np.log(1/num_features)), scale=12.0,
            name="x_bias_prior")

        # w: [num_features, num_factors]
        # F: [num_samples, num_factors]

        # TODO: here we can try using a more elaborate model (i.e. neural network)
        x_loc = tf.matmul(Z, w, transpose_b=True) + x_bias


        # dropout_rate = 0.25

        # lyr1 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        # lyr1_output = lyr1(tf.constant(Z))

        # drp1 = tf.keras.layers.Dropout(rate=dropout_rate)
        # drp1_output = drp1(lyr1_output)

        # lyr2 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        # lyr2_output = lyr2(drp1_output)

        # drp2 = tf.keras.layers.Dropout(rate=dropout_rate)
        # drp2_output = drp2(lyr2_output)

        # lyrn = tf.keras.layers.Dense(num_features)
        # x_loc = x_bias + lyrn(drp2_output)

        x_scale_hinges_diff = tf.square(tf.expand_dims(x_bias, 0) - tf.expand_dims(x_scale_hinges, -1))
        x_scale_hinges_weight_x = tf.exp(-x_scale_hinges_diff) # [scale_spline_degree, num_features]
        x_scale_hinges_weight_x = tf.clip_by_value(x_scale_hinges_weight_x, 1e-12, 1.0)
        x_scale_hinges_weight_x = x_scale_hinges_weight_x / tf.reduce_sum(x_scale_hinges_weight_x, axis=0, keepdims=True)

        # concentration = tf.nn.softplus(tf.Variable(0.0, name="concentration"))
        # rate = tf.nn.softplus(tf.Variable(0.0, name="rate"))

        x_scale_concentration_c = tf.nn.softplus(tf.Variable(
            tf.fill([scale_spline_degree], 0.0),
            name="qx_scale_concentration_c"))

        x_scale_rate_c = tf.nn.softplus(tf.Variable(
            tf.fill([scale_spline_degree], 0.0),
            name="qx_scale_rate_c"))

        x_scale_concentration_c_prior = tfd.HalfCauchy(
            loc=tf.zeros([scale_spline_degree]), scale=100.0, name="x_scale_concentration_c")

        x_scale_rate_c_prior = tfd.HalfCauchy(
            loc=tf.zeros([scale_spline_degree]), scale=100.0, name="x_scale_rate_c")

        x_scale_concentration_mix = tf.reduce_sum(
            tf.expand_dims(x_scale_concentration_c, -1) * x_scale_hinges_weight_x, axis=0)

        x_scale_rate_mix = tf.reduce_sum(
            tf.expand_dims(x_scale_rate_c, -1) * x_scale_hinges_weight_x, axis=0)

        concentration = x_scale_concentration_mix

        # mode parameterization
        # mode = tf.exp(x_scale_rate_mix)
        # rate = 1 / ((concentration + 1) * mode)
        mode = tf.exp(x_scale_rate_c[0])
        rate = 1 / ((x_scale_concentration_c[0] + 1) * mode)

        x_scale = tf.nn.softplus(tf.Variable(tf.zeros(num_features), name="x_scale"))
        x_scale_prior = tfd.InverseGamma(
            concentration=concentration,
            rate=rate,
            name="x_scale_prior")


        # x = tf.log(rnaseq_approx_likelihood_sampler_from_vars(num_samples, num_features, vars))
        x = tf.Variable(x_init, name="x", trainable=not use_point_estimates)

        x_prior = tfd.Normal(
            loc=x_loc - tf.expand_dims(sample_scales, -1),
            scale=x_scale,
            name="x_prior")

            # tf.reduce_sum(x_bias_prior.log_prob(x)) + \

        log_prior = \
            tf.reduce_sum(w_global_scale_prior.log_prob(w_global_scale)) + \
            tf.reduce_sum(w_local_scale_prior.log_prob(w_local_scale)) + \
            tf.reduce_sum(w_prior.log_prob(w)) + \
            tf.reduce_sum(x_scale_concentration_c_prior.log_prob(x_scale_concentration_c)) + \
            tf.reduce_sum(x_scale_rate_c_prior.log_prob(x_scale_rate_c)) + \
            tf.reduce_sum(x_scale_prior.log_prob(x_scale)) + \
            tf.reduce_sum(x_prior.log_prob(x))

        # log_prior = \
        #     tf.reduce_sum(x_bias_prior.log_prob(x)) + \
        #     tf.reduce_sum(x_scale_prior.log_prob(x_scale)) + \
        #     tf.reduce_sum(x_prior.log_prob(x))

        if use_point_estimates:
            log_likelihood = 0.0
        else:
            log_likelihood = rnaseq_approx_likelihood_from_vars(vars, x)

        log_posterior = log_prior + log_likelihood

        train(sess, -log_posterior, init_feed_dict, 5000, 1e-3)
        # train(sess, -log_posterior, init_feed_dict, 20, 1e-2)

        print("w_global_scale")
        print(sess.run(w_global_scale))

        print("concentration_c")
        print(sess.run(x_scale_concentration_c))

        print("rate_c")
        print(sess.run(x_scale_rate_c))

        print("x_bias")
        print(np.quantile(sess.run(x_bias), [0.0, 0.1, 0.5, 0.9, 1.0]))

        print("x_scale")
        print(np.quantile(sess.run(x_scale), [0.0, 0.1, 0.5, 0.9, 1.0]))

        print("w")
        print(np.quantile(sess.run(w), [0.0, 0.1, 0.5, 0.9, 1.0]))

        x_bias_est, w_est, x_scale_est = sess.run([x_bias, w, x_scale])
        params = {
            "x_bias": x_bias_est,
            "w": w_est,
            "x_scale": x_scale_est }

        # params = {
        #     "x_bias": sess.run(x_bias),
        #     "lyr1": sess.run(lyr1.weights),
        #     "lyr2": sess.run(lyr2.weights),
        #     "lyrn": sess.run(lyrn.weights),
        #     "x_scale": sess.run(x_scale) }

        return params


# VI
def run_probabalistic_classifier(
        sess, params, init_feed_dict, num_samples, num_features, vars,
        x_init, num_factors, sample_scales, use_point_estimates):

    # Let's try using a sampler so we can actually treat Z as discrete

    # Nope, this is not going to work, because HMC can't deal with discrete
    # variables.

    # Could do VI I guess. I suppose that's better than nothing.


    x_bias  = params["x_bias"]
    w       = params["w"]
    x_scale = params["x_scale"]

    # TODO: Let's test this with an empty model and make sure we get .5
    # probabilities.
    # w.fill(0.0)

    qZ_logits_var = tf.Variable(tf.zeros([num_samples, num_factors]))
    qZ = ed.RelaxedOneHotCategorical(
        temperature=0.5,
        logits=qZ_logits_var,
        name="qZ")

    if use_point_estimates:
        qx = ed.Deterministic(
            loc=x_init,
            name="qx")
    else:
        qx_loc_var = tf.Variable(
            x_init,
            name="qx_loc_var")
        qx_scale_var = tf.nn.softplus(
            tf.Variable(tf.zeros([num_samples, num_features])))
        qx = ed.Normal(
            loc=qx_loc_var,
            scale=qx_scale_var,
            name="qx")

    Z_prior = tfd.Normal(
        loc=0.0,
        scale=10.0,
        name="Z_prior")

    qx_loc = tf.matmul(qZ, w, transpose_b=True) + x_bias

    x_prior = tfd.Normal(
        loc=qx_loc - tf.expand_dims(sample_scales, -1),
        # loc=qx_loc,
        scale=x_scale,
        name="x_prior")

    entropy = \
        tf.reduce_sum(qZ.distribution.log_prob(qZ)) + \
        tf.reduce_sum(qx.distribution.log_prob(qx))

    log_prior = \
        tf.reduce_sum(Z_prior.log_prob(qZ)) + \
        tf.reduce_sum(x_prior.log_prob(qx))

    if use_point_estimates:
        log_likelihood = 0.0
    else:
        log_likelihood = rnaseq_approx_likelihood_from_vars(vars, qx)

    log_posterior = log_prior + log_likelihood
    elbo = log_posterior - entropy

    qZ_mean = tf.Variable(tf.zeros([num_samples, num_factors]))

    train(sess, -elbo, init_feed_dict, 2000, 1e-2)
    # train(sess, -elbo, init_feed_dict, 200, 1e-2)

    accum_op = tf.assign(qZ_mean, qZ_mean + qZ)
    niter = 100
    for _ in range(niter):
        sess.run(accum_op)
    qZ_mean_val = sess.run(qZ_mean / niter)

    # return sess.run(qZ.distribution.mode())
    print(sess.run(qZ_logits_var))
    # return sess.run(tf.nn.softmax(qZ_logits_var, axis=1)), w, x_bias
    return qZ_mean_val, w, x_bias


# Maximum posterior
if False:
    def run_probabalistic_classifier(
            sess, params, init_feed_dict, num_samples, num_features, vars,
            x_init, num_factors, sample_scales, use_point_estimates):

        x_bias  = params["x_bias"]
        w       = params["w"]
        x_scale = params["x_scale"]

        Z = tf.nn.softmax(
            tf.Variable(tf.zeros([num_samples, num_factors]), name="Z"),
            axis=1)
        Z_prior = tfd.Normal(
            loc=0.0,
            scale=10.0,
            name="Z_prior")

        x_loc = tf.matmul(Z, w, transpose_b=True) + x_bias

        # dropout_rate = 0.25

        # lyr1 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        # lyr1_output = lyr1(Z)

        # drp1 = tf.keras.layers.Dropout(rate=dropout_rate)
        # drp1_output = drp1(lyr1_output)

        # lyr2 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        # lyr2_output = lyr2(drp1_output)

        # drp2 = tf.keras.layers.Dropout(rate=dropout_rate)
        # drp2_output = drp2(lyr2_output)

        # lyrn = tf.keras.layers.Dense(num_features)
        # x_loc = x_bias + lyrn(drp2_output)

        x = tf.Variable(x_init, name="x", trainable=not use_point_estimates)
        # x = tf.log(rnaseq_approx_likelihood_sampler_from_vars(num_samples, num_features, vars))

        x_prior = tfd.Normal(
            # loc=x_loc - tf.expand_dims(sample_scales, -1),
            loc=x_loc,
            scale=x_scale,
            name="x_prior")

        Z_prior_log_prob = tf.reduce_sum(Z_prior.log_prob(Z))
        x_prior_log_prob = tf.reduce_sum(x_prior.log_prob(x))
        log_prior = Z_prior_log_prob + x_prior_log_prob

        log_prior = tf.Print(log_prior, [Z_prior_log_prob, x_prior_log_prob], "log_prior parts")

        # log_prior = \
        #     tf.reduce_sum(Z_prior.log_prob(Z)) + \
        #     tf.reduce_sum(x_prior.log_prob(x))

        if use_point_estimates:
            log_likelihood = 0.0
        else:
            log_likelihood = rnaseq_approx_likelihood_from_vars(vars, x)

        log_prior = tf.Print(log_prior, [log_prior], "log_prior")
        log_likelihood = tf.Print(log_likelihood, [log_likelihood], "log_likelihood")

        log_posterior = log_prior + log_likelihood

        # sess.run(tf.assign(lyr1.weights[0], params["lyr1"][0]))
        # sess.run(tf.assign(lyr1.weights[1], params["lyr1"][1]))
        # sess.run(tf.assign(lyr2.weights[0], params["lyr2"][0]))
        # sess.run(tf.assign(lyr2.weights[1], params["lyr2"][1]))
        # sess.run(tf.assign(lyrn.weights[0], params["lyrn"][0]))
        # sess.run(tf.assign(lyrn.weights[1], params["lyrn"][1]))

        # train(sess, -log_posterior, init_feed_dict, 3000, 1e-2)
        train(sess, -log_posterior, init_feed_dict, 2000, 1e-3)

        return sess.run(Z), w, x_bias


# Importance sampler
if False:
    def run_probabalistic_classifier(
            sess, params, init_feed_dict, num_samples, num_features, vars,
            x_init, num_factors, sample_scales, use_point_estimates):

        Z = ed.OneHotCategorical(
            logits=tf.zeros([num_samples, num_factors]),
            name="Z")

        x_bias  = params["x_bias"]
        w       = params["w"]
        x_scale = params["x_scale"]

        x_loc = tf.matmul(Z, w, transpose_b=True) + x_bias

        # dropout_rate = 0.25

        # lyr1 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        # lyr1_output = lyr1(Z)

        # drp1 = tf.keras.layers.Dropout(rate=dropout_rate)
        # drp1_output = drp1(lyr1_output)

        # lyr2 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        # lyr2_output = lyr2(drp1_output)

        # drp2 = tf.keras.layers.Dropout(rate=dropout_rate)
        # drp2_output = drp2(lyr2_output)

        # lyrn = tf.keras.layers.Dense(num_features)
        # x_loc = x_bias + lyrn(drp2_output)

        # x = tf.Variable(x_init, name="x", trainable=not use_point_estimates)
        x = tf.log(rnaseq_approx_likelihood_sampler_from_vars(num_samples, num_features, vars))

        x_prior = tfd.Normal(
            # loc=x_loc - tf.expand_dims(sample_scales, -1),
            loc=x_loc,
            scale=x_scale,
            name="x_prior")

        # log_prior = tf.Print(log_prior, [Z_prior_log_prob, x_prior_log_prob], "log_prior parts")

        # log_prior = \
        #     tf.reduce_sum(Z_prior.log_prob(Z)) + \
        #     tf.reduce_sum(x_prior.log_prob(x))

        # if use_point_estimates:
        #     log_likelihood = 0.0
        # else:
        #     log_likelihood = rnaseq_approx_likelihood_from_vars(vars, x)

        # log_prior = tf.Print(log_prior, [log_prior], "log_prior")
        # log_likelihood = tf.Print(log_likelihood, [log_likelihood], "log_likelihood")

        # log_posterior = log_prior + log_likelihood

        # sess.run(tf.assign(lyr1.weights[0], params["lyr1"][0]))
        # sess.run(tf.assign(lyr1.weights[1], params["lyr1"][1]))
        # sess.run(tf.assign(lyr2.weights[0], params["lyr2"][0]))
        # sess.run(tf.assign(lyr2.weights[1], params["lyr2"][1]))
        # sess.run(tf.assign(lyrn.weights[0], params["lyrn"][0]))
        # sess.run(tf.assign(lyrn.weights[1], params["lyrn"][1]))

        # train(sess, -log_posterior, init_feed_dict, 3000, 1e-2)
        train(sess, -log_posterior, init_feed_dict, 2000, 1e-3)

        return sess.run(Z), w, x_bias

