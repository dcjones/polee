
"""
Simple bayesian linear regression.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import sys
from tensorflow_probability import edward2 as ed
from polee_approx_likelihood import *
from polee_training import *


scale_spline_degree = 10


class GammaNonNan(tfd.Gamma):
    def __init__(self, concentration, rate):
        super(GammaNonNan, self).__init__(concentration, rate)

    def log_prob(self, x):
        # return super(GammaNonNan, self).log_prob(tf.maximum(x, 0.0))
        return super(GammaNonNan, self).log_prob(tf.maximum(x, 1e-6))

"""
Define model for linear regression.
    * `num_factors`: Number of factors
    * `num_features`: Dimensionality
    * `F`: 0/1 design matrix of shape [num_samples, num_factors]
"""
def linear_regression_model(
        num_factors, num_features, F,
        x_bias_loc, x_bias_scale, x_scale_hinges, x_scale_hinge_weights):

    x_bias = ed.Normal(
        loc=tf.fill([num_features], np.float32(x_bias_loc)),
        scale=np.float32(x_bias_scale),
        name="x_bias")

    x_scale_hinges_diff = tf.square(tf.expand_dims(x_bias, 0) - tf.expand_dims(x_scale_hinges, -1))
    # x_scale_hinges_diff = tf.clip_by_value(x_scale_hinges_diff, -100.0, 100.0)
    # x_scale_hinges_diff = tf.Print(
    #     x_scale_hinges_diff,
    #     [tf.reduce_min(x_scale_hinges_diff), tf.reduce_max(x_scale_hinges_diff)],
    #     "x_scale_hinges_diff")

    # TODO: just replace this with softmax
    x_scale_hinges_weight_w = tf.exp(-x_scale_hinges_diff / tf.square(tf.expand_dims(x_scale_hinge_weights, -1))) # [scale_spline_degree, num_features]
    x_scale_hinges_weight_w = tf.clip_by_value(x_scale_hinges_weight_w, 1e-10, 1.0)
    # x_scale_hinges_weight_w = tf.Print(
    #     x_scale_hinges_weight_w,
    #     [tf.reduce_min(x_scale_hinges_weight_w), tf.reduce_max(x_scale_hinges_weight_w)],
    #     "x_scale_hinges_weight_w")
    x_scale_hinges_weight_w = x_scale_hinges_weight_w / tf.reduce_sum(x_scale_hinges_weight_w, axis=0, keepdims=True)

    # x_scale_hinges_weight_x = tf.exp(-x_scale_hinges_diff / tf.square(tf.expand_dims(x_scale_hinge_weights, -1))) # [scale_spline_degree, num_features]
    # x_scale_hinges_weight_x = tf.clip_by_value(x_scale_hinges_weight_x, 1e-10, 1.0)
    # # x_scale_hinges_weight_x = tf.Print(
    # #     x_scale_hinges_weight_x,
    # #     [tf.reduce_min(x_scale_hinges_weight_x), tf.reduce_max(x_scale_hinges_weight_x)],
    # #     "x_scale_hinges_xeight_x")
    # x_scale_hinges_weight_x = x_scale_hinges_weight_x / tf.reduce_sum(x_scale_hinges_weight_x, axis=0, keepdims=True)

    # w
    # -

    # horseshoe prior
    tau = ed.HalfCauchy(loc=tf.zeros([scale_spline_degree]), scale=10.0, name="tau")

    w_scale = ed.HalfCauchy(loc=tf.zeros([num_features]), scale=10.0, name="w_scale")
    # w_scale = ed.LogNormal(loc=tf.zeros([num_features]), scale=2.0, name="w_scale")

    # [num_features]
    tau_mix = tf.reduce_sum(
        tf.expand_dims(tau, -1) * x_scale_hinges_weight_w, axis=0)

    # w = ed.StudentT(
    #     df=0.1,

    # w_df = ed.HalfCauchy(loc=0.0, scale=1.0, name="w_df")
    w_df = ed.Normal(loc=[0.0, 0.0, 0.0], scale=[10.0, 10.0, 10.0], name="w_df")

    # w = ed.StudentT(
    #     # df=tf.sigmoid(w_df),
    #     # df=w_df,
    #     # df=tf.expand_dims(w_scale, -1),
    #     df=1.0,
    # # w = ed.Normal(
    #     loc=0.0,
    #     scale=tf.expand_dims(tau_mix, -1),
    #     # scale=tf.ones([num_features, num_factors]) * tau[0],
    #     # scale=tf.fill([num_features, num_factors], 10.0),
    #     # scale=tf.ones([1, num_factors]) * tf.expand_dims(w_scale * tau[0], -1),
    #     # scale=tf.ones([1, num_factors]) * tf.expand_dims(w_scale * tau_mix, -1),
    #     # scale=tf.fill([num_features, num_factors], 0.06971968),
    #     name="w")

    # This (one of) the prior(s) described in Lewin, et al 2007

    # print(tf.ones([num_features, num_factors, 1]) * w_df)

    # print(tfd.Categorical(logits=tf.ones([num_features, num_factors, 1]) * w_df))

    # print(tfd.Normal(
    #     loc=tf.zeros([num_features, num_factors]),
    #     scale=tf.ones([num_features, num_factors]) * 0.01))

    # print(tfd.TransformedDistribution(
    #     distribution=tfd.Gamma(
    #         concentration=tf.ones([num_features, num_factors]) * 1.0,
    #         rate=tf.ones([num_features, num_factors]) * tau[0]),
    #     bijector=tfp.bijectors.Affine(0.0, -1.0)))

    # print(tfd.Gamma(
    #     concentration=tf.ones([num_features, num_factors]) * 1.0,
    #     rate=tf.ones([num_features, num_factors]) * tau[0]))

    # w = ed.StudentT(
    #     df=tf.exp(w_df[0]),
    # w = ed.Normal(
    #     loc=tf.zeros([num_features, num_factors]),
    #     # scale=tf.ones([num_features, num_factors]) * 0.1,
    #     scale=tf.ones([num_features, num_factors]) * tau[0],
    #     # scale=tf.ones([1, num_factors]) * tf.expand_dims(w_scale * tau_mix, -1),
    #     # scale=tf.expand_dims(tau_mix, -1),
    #     name="w")

    w = ed.Normal(
        loc=tf.zeros([num_features, num_factors]),
        scale=tf.expand_dims(tau_mix, -1),
        name="w")

    # w = ed.Mixture(
    #     cat=tfd.Categorical(logits=tf.ones([num_features, num_factors, 1]) * w_df),
    #     # cat=tfd.Categorical(probs=tf.ones([num_features, num_factors, 1]) * [0.8, 0.1, 0.1]),
    #     components=[
    #         # null distribution
    #         tfd.Normal(
    #             loc=tf.zeros([num_features, num_factors]),
    #             scale=tf.ones([num_features, num_factors]) * 0.05),
    #         # downreg distribution
    #         tfd.TransformedDistribution(
    #             distribution=GammaNonNan(
    #                 concentration=tf.ones([num_features, num_factors]) * 3.0,
    #                 rate=tf.ones([num_features, num_factors]) * tau[0]),
    #                 # rate=tf.ones([num_features, num_factors]) * 1.0),
    #             bijector=tfp.bijectors.AffineScalar(scale=-1.0)),
    #         # GammaNonNan(
    #         #     concentration=tf.ones([num_features, num_factors]) * 1.5,
    #         #     # rate=tf.ones([num_features, num_factors]) * tau[0])],
    #         #     rate=tf.ones([num_features, num_factors]) * 1.0),
    #         # upreg distribution
    #         GammaNonNan(
    #             concentration=tf.ones([num_features, num_factors]) * 3.0,
    #             rate=tf.ones([num_features, num_factors]) * tau[0])],
    #             # rate=tf.ones([num_features, num_factors]) * 1.0)],
    #     name="w")

    # TODO: The problem here is that evaluating probability of a negative number
    # on the Gamma will give a NaN. I guess I need to make a custom distribution.

    # We can't do horseshoe like this because w=0.0 has non-finite probability.
    # w = ed.Horseshoe(
    #     # scale=tf.expand_dims(tau_mix, -1),
    #     scale=tf.ones([num_features, num_factors]),
    #     name="w")

    # w = ed.Mixture(
    #     # mixture_distribution=tfd.Categorical(logits=[tf.log(w_df)]),
    #     # cat=tfd.Categorical(logits=tf.ones([num_features, num_factors, 1]) * [-1.0, 1.0]),
    #     # cat=tfd.Categorical(logits=tf.ones([num_features, num_factors, 1]) * [0.0, tf.log(w_df)]),
    #     cat=tfd.Categorical(probs=tf.ones([num_features, num_factors, 1]) * [0.2, 0.9]),
    #     # cat=tfd.Categorical(logits=tf.ones([num_features, num_factors, 1]) * w_df),
    #     # tf.expand_dims(tf.expand_dims([-1.0, 1.0], axis=0), axis=0)),
    #     components=[
    #         tfd.StudentT(
    #             df=1.0,
    #             loc=0.0,
    #             # scale=tf.expand_dims(tau_mix, -1)),
    #             # scale=tf.ones([num_features, num_factors]) * tau[0]),
    #             scale=tf.ones([num_features, num_factors]) * 2.0),
    #         # tfd.Normal(
    #         #     loc=tf.zeros([num_features, num_factors]),
    #         #     scale=tf.ones([num_features, num_factors]) * 1.0),
    #         tfd.Normal(
    #             loc=tf.zeros([num_features, num_factors]),
    #             # scale=tf.expand_dims(tau_mix, -1))],
    #             scale=tf.ones([num_features, num_factors]) * 0.1)],
    #     name="w")

    # w = ed.Mixture(
    #     # mixture_distribution=tfd.Categorical(logits=[tf.log(w_df)]),
    #     # cat=tfd.Categorical(logits=tf.ones([num_features, num_factors, 1]) * [-1.0, 1.0]),
    #     cat=tfd.Categorical(logits=tf.ones([num_features, num_factors, 1]) * w_df),
    #     # tf.expand_dims(tf.expand_dims([-1.0, 1.0], axis=0), axis=0)),
    #     components=[
    #         tfd.StudentT(
    #             df=0.1,
    #             loc=tf.zeros([num_features, num_factors]),
    #             # scale=tf.expand_dims(tau_mix, -1)),
    #             scale=tf.ones([num_features, num_factors]) * tau[0]),
    #         tfd.StudentT(
    #             df=1.0,
    #             loc=tf.zeros([num_features, num_factors]),
    #             # scale=tf.expand_dims(tau_mix, -1))],
    #             scale=tf.ones([num_features, num_factors]) * tau[0])],
    #     name="w")

    # x
    # -

    x_loc = tf.identity(
        tf.matmul(F, w, transpose_b=True) + x_bias,
        name="x_loc") # [num_samples, num_features]

    print(x_loc)

    # x_scale_hinges_diff = tf.square(tf.expand_dims(x_loc, 0) - tf.expand_dims(tf.expand_dims(x_scale_hinges, -1), -1))
    # x_scale_hinges_weight_x = tf.exp(-x_scale_hinges_diff / tf.square(tf.expand_dims(tf.expand_dims(x_scale_hinge_weights, -1), -1))) # [scale_spline_degree, num_features]
    # x_scale_hinges_weight_x = tf.clip_by_value(x_scale_hinges_weight_x, 1e-10, 1.0)
    # x_scale_hinges_weight_x = x_scale_hinges_weight_x / tf.reduce_sum(x_scale_hinges_weight_x, axis=0, keepdims=True)

    x_scale_hinges_diff = tf.square(tf.expand_dims(x_bias, 0) - tf.expand_dims(x_scale_hinges, -1))
    x_scale_hinges_weight_x = tf.exp(-x_scale_hinges_diff / (0.5 * tf.square(tf.expand_dims(x_scale_hinge_weights, -1)))) # [scale_spline_degree, num_features]
    x_scale_hinges_weight_x = tf.clip_by_value(x_scale_hinges_weight_x, 1e-10, 1.0)
    x_scale_hinges_weight_x = x_scale_hinges_weight_x / tf.reduce_sum(x_scale_hinges_weight_x, axis=0, keepdims=True)

    x_scale_concentration_c = ed.Normal(
        loc=tf.fill([scale_spline_degree], 0.0), scale=100.0, name="x_scale_concentration_c")

    x_scale_rate_c = ed.Normal(
        loc=tf.fill([scale_spline_degree], 0.0), scale=100.0, name="x_scale_rate_c")

    x_scale_concentration_mix = tf.reduce_sum(
        tf.expand_dims(x_scale_concentration_c, -1) * x_scale_hinges_weight_x, axis=0)
        # tf.expand_dims(tf.expand_dims(x_scale_concentration_c, -1), -1) * x_scale_hinges_weight_x, axis=0)

    x_scale_rate_mix = tf.reduce_sum(
        tf.expand_dims(x_scale_rate_c, -1) * x_scale_hinges_weight_x, axis=0)
        # tf.expand_dims(tf.expand_dims(x_scale_rate_c, -1), -1) * x_scale_hinges_weight_x, axis=0)

    x_scale_scale = ed.HalfCauchy(
        loc=0.0,
        scale=10.0,
        name="x_scale_scale")

    # TODO: OK!!! Here's the issue! X scale is not properly set!
    # How can I deal with this?

    # x_scale = ed.TransformedDistribution(
    #     distribution=tfd.Normal(
    #     # distribution=tfd.Cauchy(
    #         loc=x_scale_loc_mix,
    #         # loc=tf.ones([num_features]) * x_scale_c[0],
    #         # loc=tf.fill([num_features], -1.0),
    #         # scale=0.4),
    #         scale=x_scale_scale),
    #     bijector=tfp.bijectors.Exp(),
    #     name="x_scale")

    # x_scale = ed.TransformedDistribution(
    #     distribution=tfd.Normal(
    #         loc=x_scale_concentration_mix,
    #         # scale=x_scale_rate_mix),
    #         scale=0.5),
    #     bijector=tfp.bijectors.Exp(),
    #     name="x_scale")



    # This does ok

    mode = tf.exp(x_scale_rate_mix)
    # mode = tf.exp(x_scale_rate_c[0])

    # sd = tf.exp(x_scale_rate_mix)
    # sd = tf.ones([num_features]) * x_scale_rate_c[0]
    # sd = tf.ones([num_features]) * 0.2

    # inverse-gamma parameters
    # concentration = tf.exp(x_scale_concentration_mix)
    concentration = tf.exp(x_scale_concentration_c[0])
    # concentration = 10.0
    # rate = 1 / ((concentration + 1) * mode)
    rate = (concentration + 1) * mode

    # gamma parameters
    # rate = (mode + tf.sqrt(mode**2 + 4*sd**2)) / (2 * sd**2)
    # concentration = 1 + mode * rate

    # x_scale = ed.InverseGamma(
    #     concentration=concentration,
    #     rate=rate,
    #     name="x_scale")

    x_scale = ed.InverseGamma(
        concentration=concentration,
        rate=rate,
        name="x_scale")

    # Try doing precision istead of sd

    # x_scale = ed.HalfCauchy(
    #     loc=0.0,
    #     # scale=tf.exp(x_scale_rate_mix),
    #     scale=tf.ones(x_loc.shape) * tf.exp(x_scale_rate_c[0]),
    #     name="x_scale")

    x = ed.Normal(
        loc=x_loc,
        scale=x_scale,
        # scale=1.0/x_scale,
        name="x")

    return tau, w_scale, w_df, w, x_bias, x_scale_concentration_c, x_scale_rate_c, x_scale_scale, x_scale, x
    # return tau, w_scale, w, x_bias, x_scale_c, x_scale, x


"""
Variational model for linear regression, to be paired with `linear_regression_model`.
"""
def linear_regression_variational_model(
        qtau_loc_var, qtau_scale_var,
        qw_scale_loc_var, qw_scale_scale_var,
        qw_df_loc_var,
        qw_loc_var, qw_scale_var,
        qx_bias_loc_var, qx_bias_scale_var,
        qx_scale_concentration_c_loc_var, qx_scale_concentration_c_scale_var,
        qx_scale_rate_c_loc_var, qx_scale_rate_c_scale_var,
        qx_scale_scale_loc_var,
        qx_scale_loc_var, qx_scale_scale_var,
        qx_loc_var, qx_scale_var,
        use_point_estimates):

    qtau = ed.Deterministic(
        # loc=tf.exp(qtau_loc_var),
        loc=tf.nn.softplus(qtau_loc_var),
        name="qtau")

    qw_df = ed.Deterministic(
        # loc=tf.exp(qw_df_loc_var),
        # loc=tf.nn.softplus(qw_df_loc_var),
        loc=qw_df_loc_var,
        name="qw_df")

    qw_scale = ed.LogNormal(
        loc=qw_scale_loc_var,
        scale=qw_scale_scale_var,
        name="qw_scale")

    # qw_scale = ed.Gamma(
    #     concentration=tf.nn.softplus(qw_scale_loc_var),
    #     rate=qw_scale_scale_var,
    #     name="qw_scale")

    # qw_scale = ed.Deterministic(
    #     loc=qw_scale_loc_var,
    #     name="qw_scale")

    qw = ed.Normal(
        loc=qw_loc_var,
        scale=qw_scale_var,
        name="qw")

    qx_bias = ed.Normal(
        loc=qx_bias_loc_var,
        scale=qx_bias_scale_var,
        name="qx_bias")

    qx_scale_concentration_c = ed.Deterministic(
        loc=qx_scale_concentration_c_loc_var,
        name="qx_scale_concentration_c")

    qx_scale_rate_c = ed.Deterministic(
        loc=qx_scale_rate_c_loc_var,
        name="qx_scale_rate_c")

    qx_scale_scale = ed.Deterministic(
        loc=qx_scale_scale_loc_var,
        # loc=tf.Print(qx_scale_scale_loc_var, [qx_scale_scale_loc_var], "qx_scale_scale_loc_var"),
        name="qx_scale_scale")

    qx_scale = ed.LogNormal(
        loc=qx_scale_loc_var,
        scale=qx_scale_scale_var,
        name="qx_scale")

    # qx_scale = ed.InverseGamma(
    #     concentration=tf.nn.softplus(qx_scale_loc_var),
    #     rate=qx_scale_scale_var,
    #     name="qx_scale")

    # qx_scale = ed.Gamma(
    #     concentration=tf.nn.softplus(qx_scale_loc_var),
    #     rate=qx_scale_scale_var,
    #     name="qx_scale")

    # qx_scale = ed.Deterministic(
    #     loc=qx_scale_loc_var,
    #     name="qx_scale")

    if use_point_estimates:
        qx = ed.Deterministic(loc=qx_loc_var, name="qx")
    else:
        qx = ed.Normal(
            loc=qx_loc_var,
            scale=qx_scale_var,
            name="qx")

    return qtau, qw_scale, qw_df, qw, qx_bias, qx_scale_concentration_c, qx_scale_rate_c, qx_scale_scale, qx_scale, qx
    # return qtau, qw_scale, qw, qx_bias, qx_scale_c, qx_scale, qx



"""
Set up a linear regression model for variational inference, returning
"""
def linear_regression_inference(
        init_feed_dict, F, x_init, make_likelihood,
        x_bias_mu0, x_bias_sigma0, x_scale_hinges, x_scale_hinge_weights, sample_scales,
        use_point_estimates, sess):

    num_samples = int(F.shape[0])
    num_factors = int(F.shape[1])
    num_features = int(x_init.shape[1])

    log_joint = ed.make_log_joint_fn(
        lambda: linear_regression_model
            (num_factors, num_features, F, x_bias_mu0, x_bias_sigma0, x_scale_hinges, x_scale_hinge_weights))

    qtau_loc_var = tf.Variable(tf.fill([scale_spline_degree], -2.0), name="qtau_loc_var")
    # qtau_loc_var = tf.Variable([-2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], name="qtau_loc_var")
    qtau_scale_var = tf.nn.softplus(tf.Variable(tf.zeros([scale_spline_degree]), name="qtau_scale_var"))

    qw_scale_loc_var = tf.Variable(
        tf.zeros([num_features]),
        name="qw_scale_loc_var")
    qw_scale_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_features], -1.0),
        name="qw_scale_loc_var"))

    qw_df_loc_var = tf.Variable(
        [2.0, -2.0, -2.0],
        name="qw_df_loc_var")

    qw_loc_var = tf.Variable(
        tf.zeros([num_features, num_factors]),
        name="qx_loc_var")
    qw_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_features, num_factors], -1.0),
        name="qx_scale_var"))

    qw_loc_var = tf.check_numerics(qw_loc_var, "Non-finite value in qw_loc_var")
    qw_scale_var = tf.check_numerics(qw_scale_var, "Non-finite value in qw_scale_var")

    qx_bias_loc_var = tf.Variable(
        tf.reduce_mean(x_init, axis=0),
        name="qx_bias_loc_var")
    qx_bias_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_features], -1.0),
        name="qx_bias_scale_var"))

    # qx_scale_c_loc_var = tf.Variable(
    #     tf.zeros([scale_spline_degree]),
    #     name="qx_scale_c_loc_var")

    qx_scale_concentration_c_loc_var = tf.Variable(
        tf.fill([scale_spline_degree], 1.0),
        name="qx_scale_concentration_c_loc_var")

    qx_scale_concentration_c_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([scale_spline_degree], -1.0),
        name="qx_scale_concentration_c_scale_var"))

    qx_scale_rate_c_loc_var = tf.Variable(
        tf.fill([scale_spline_degree], 1.0),
        name="qx_scale_rate_c_loc_var")

    qx_scale_rate_c_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([scale_spline_degree], -1.0),
        name="qx_scale_rate_c_scale_var"))

    qx_scale_scale_loc_var = tf.nn.softplus(tf.Variable(
        0.0, name="qx_scale_scale_loc_var"))

    # qx_scale_loc_var = tf.Variable(
    #     tf.fill([num_samples, num_features], 3.0),
    #     name="qx_scale_loc_var")
    # qx_scale_scale_var = tf.nn.softplus(tf.Variable(
    #     tf.fill([num_samples, num_features], -1.0),
    #     name="qx_scale_scale_var"))
    qx_scale_loc_var = tf.Variable(
        tf.fill([num_features], 3.0),
        name="qx_scale_loc_var")
    qx_scale_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_features], -1.0),
        name="qx_scale_scale_var"))

    qx_loc_var = tf.Variable(
        x_init,
        name="qx_loc_var",
        trainable=not use_point_estimates)

    qx_scale_var = tf.nn.softplus(tf.Variable(
        tf.fill([num_samples, num_features], 0.0),
        name="qx_scale_var"))

    # qx_loc_var = tf.Variable(
    #     x_init,
    #     name="qx_loc_var",
    #     trainable=False)

    # qx_scale_var_ = tf.Variable(
    #     tf.fill([num_samples, num_features], -5.0),
    #     name="qx_scale_var",
    #     trainable=False)
    # qx_scale_var = tf.nn.softplus(qx_scale_var_)

    # qtau, qw_scale, qw, qx_bias, qx_scale_c, qx_scale, qx = \
    qtau, qw_scale, qw_df, qw, qx_bias, qx_scale_concentration_c, qx_scale_rate_c, qx_scale_scale, qx_scale, qx = \
        linear_regression_variational_model(
            qtau_loc_var, qtau_scale_var,
            qw_scale_loc_var, qw_scale_scale_var,
            qw_df_loc_var,
            qw_loc_var, qw_scale_var,
            qx_bias_loc_var, qx_bias_scale_var,
            qx_scale_concentration_c_loc_var, qx_scale_concentration_c_scale_var,
            qx_scale_rate_c_loc_var, qx_scale_rate_c_scale_var,
            qx_scale_scale_loc_var,
            qx_scale_loc_var, qx_scale_scale_var,
            qx_loc_var, qx_scale_var,
            use_point_estimates)

    log_prior = log_joint(
        tau=qtau,
        w_scale=qw_scale,
        w_df=qw_df,
        w=qw,
        x_bias=qx_bias,
        x_scale_concentration_c=qx_scale_concentration_c,
        x_scale_rate_c=qx_scale_rate_c,
        x_scale_scale=qx_scale_scale,
        x_scale=qx_scale,
        x=qx)

    variational_log_joint = ed.make_log_joint_fn(
        lambda: linear_regression_variational_model(
            qtau_loc_var, qtau_scale_var,
            qw_scale_loc_var, qw_scale_scale_var,
            qw_df_loc_var,
            qw_loc_var, qw_scale_var,
            qx_bias_loc_var, qx_bias_scale_var,
            qx_scale_concentration_c_loc_var, qx_scale_concentration_c_scale_var,
            qx_scale_rate_c_loc_var, qx_scale_rate_c_scale_var,
            qx_scale_scale_loc_var,
            qx_scale_loc_var, qx_scale_scale_var,
            qx_loc_var, qx_scale_var,
            use_point_estimates))

    entropy = variational_log_joint(
        qtau=qtau,
        qw_scale=qw_scale,
        qw_df=qw_df,
        qw=qw,
        qx_bias=qx_bias,
        qx_scale_scale=qx_scale_scale,
        qx_scale=qx_scale,
        qx_scale_concentration_c=qx_scale_concentration_c,
        qx_scale_rate_c=qx_scale_rate_c,
        qx=qx)

    log_likelihood = make_likelihood(qx)

    scale_penalty = tf.reduce_sum(tfd.Normal(
        loc=sample_scales,
        scale=1e-3).log_prob(tf.log(tf.reduce_sum(tf.exp(qx), axis=1))))

    # x_bias_penalty = tf.reduce_sum(tfd.Normal(
    #     loc=0.0,
    #     scale=1e-3).log_prob(tf.log(tf.reduce_sum(tf.exp(qx_bias)))))

    elbo = log_prior + log_likelihood - entropy + scale_penalty
    # elbo = log_prior + log_likelihood - entropy
    elbo = tf.check_numerics(elbo, "Non-finite ELBO value")

    # elbo = log_prior + log_likelihood - entropy

    # elbo = log_prior + log_likelihood - entropy

    if sess is None:
        sess = tf.Session()

    # train(sess, -elbo, init_feed_dict, 20000, 1e-3, decay_rate=0.999)
    # train(sess, -elbo, init_feed_dict, 30000, 1e-3, decay_rate=0.9995)
    # train(sess, -elbo, init_feed_dict, 30000, 1e-3, decay_rate=1.0)
    # train(sess, -elbo, init_feed_dict, 80000, 1e-4, decay_rate=1.0)
    # train(sess, -elbo, init_feed_dict, 20000, 1e-3, decay_rate=0.999)
    train(sess, -elbo, init_feed_dict, 20000, 1e-2, decay_rate=0.999)

    # train(sess, -elbo, init_feed_dict, 20000, 1e-3, decay_rate=1.0)
    # train(sess, -elbo, init_feed_dict, 40000, 1e-1, decay_rate=0.999)
    # train(sess, -elbo, init_feed_dict, 20000, 1e-3, decay_rate=0.999)
    # train(sess, -elbo, init_feed_dict, 80000, 1e-4, decay_rate=1.0)
    # train(sess, -elbo, init_feed_dict, 80000, 1e-2, decay_rate=0.999)

    # train(sess, -elbo, init_feed_dict, 20000, 1e-3, decay_rate=1.0,
    #     initialized_vars=set(tf.all_variables()),
    #     var_list=tf.trainable_variables() + [qx_loc_var, qx_scale_var_])

    print("tau")
    print(sess.run(qtau))

    print("x_scale_concentration_c")
    print(sess.run(tf.exp(qx_scale_concentration_c)))

    print("x_scale_rate_c")
    print(sess.run(tf.exp(qx_scale_rate_c)))

    print("x_scale_scale")
    print(sess.run(qx_scale_scale))

    print("x_bias quantile")
    print(np.quantile(sess.run(qx_bias), [0.0, 0.1, 0.5, 0.9, 1.0]))

    print("x_bias sum")
    print(sess.run(tf.reduce_sum(tf.exp(qx_bias))))

    print("x sums")
    print(sess.run(tf.reduce_sum(tf.exp(qx), axis=1)))

    print("x quantiles")
    for i in range(num_samples):
        print(np.quantile(sess.run(qx[i,:]), [0.0, 0.1, 0.5, 0.9, 1.0]))

    print("w_df")
    print(sess.run(qw_df))
    print(sess.run(tf.nn.softmax(qw_df)))

    return (
        sess.run(qx.distribution.mean()),
        sess.run(qw.distribution.mean()),
        sess.run(qw.distribution.stddev()),
        sess.run(qx_bias.distribution.mean()),
        sess.run(qx_scale.distribution.mean()))


"""
Run variational inference on transcript expression linear regression.
"""
def estimate_transcript_linear_regression(
        init_feed_dict, vars, x_init, F_arr,
        use_point_estimates, sess=None):

    F = tf.constant(F_arr, dtype=tf.float32)
    num_features = x_init.shape[1]

    # TODO: trying different initalization
    x_init = np.repeat(np.mean(x_init, axis=0, keepdims=True), repeats=x_init.shape[0], axis=0)

    x_bias_mu0 = np.log(1/num_features)
    x_bias_sigma0 = 16.0

    if use_point_estimates:
        make_likelihood = lambda qx: 0.0
    else:
        make_likelihood = lambda qx: rnaseq_approx_likelihood_from_vars(vars, qx)

    return linear_regression_inference(
        init_feed_dict, F, x_init, make_likelihood,
        x_bias_mu0, x_bias_sigma0, use_point_estimates, sess)


"""
Run variational inference on feature log-ratio linear regression.
"""
def estimate_feature_linear_regression(
        init_feed_dict, feature_loc, feature_scale, x0_log, F_arr, sample_scales,
        use_point_estimates, sess=None):

    # don't need this since we already used the transcript expression
    # likelihood approximation to approximate splicing likelihood.
    init_feed_dict.clear()

    num_samples = feature_loc.shape[0]
    num_features = feature_scale.shape[1]

    feature_likelihood = ed.Normal(
        loc=feature_loc,
        scale=feature_scale,
        name="feature_likelihood")

    if use_point_estimates:
        make_likelihood = lambda qx: 0.0
    else:
        make_likelihood = lambda qx: tf.reduce_sum(feature_likelihood.distribution.log_prob(
            tf.log(tf.nn.softmax(qx, axis=1))))
    # make_likelihood = lambda qx: tf.reduce_sum(feature_likelihood.distribution.log_prob(qx))

    F = tf.constant(F_arr, dtype=tf.float32)

    x_init = tf.log(tf.nn.softmax(feature_loc, axis=1))
    # x_init = feature_loc

    # choose equally spaced points for piecewise variance function
    x_init_exp = np.exp(feature_loc)
    x_init_mean = np.mean(np.log(x_init_exp / np.sum(x_init_exp, axis=1, keepdims=True)), axis=0)

    x_scale_hinges = []
    x_init_mean_min = np.min(x_init_mean)
    x_init_mean_max = np.max(x_init_mean)
    d = (x_init_mean_max - x_init_mean_min) / (scale_spline_degree+1)
    for i in range(scale_spline_degree):
        x_scale_hinges.append(x_init_mean_min + (i+1)*d)
    x_scale_hinge_weights = np.ones(scale_spline_degree, dtype=np.float32)

    # x_scale_hinges = np.float32(np.quantile(
    #     x_init_mean,
    #     np.linspace(0.0, 1.0, scale_spline_degree+1, endpoint=False)[1:]))

    # x_scale_hinge_weights = np.copy(x_scale_hinges)
    # for i in range(scale_spline_degree):
    #     if i == 0:
    #         x_scale_hinge_weights[i] = x_scale_hinges[i+1] - x_scale_hinges[i]
    #     elif i == scale_spline_degree - 1:
    #         x_scale_hinge_weights[i] = x_scale_hinges[i] - x_scale_hinges[i-1]
    #     else:
    #         x_scale_hinge_weights[i] = \
    #             (x_scale_hinges[i+1] - x_scale_hinges[i])/2 + \
    #             (x_scale_hinges[i] - x_scale_hinges[i-1])/2

    # print("mean quantiles")
    # print(np.quantile(x_init_mean, [0.0, 0.1, 0.5, 0.9, 1.0]))

    print("hinges")
    print(x_scale_hinges)

    print("hinge weights")
    print(x_scale_hinge_weights)

    x_bias_mu0 = np.log(1/num_features)
    x_bias_sigma0 = 16.0

    if sess is None:
        sess = tf.Session()

    return linear_regression_inference(
        init_feed_dict, F, x_init, make_likelihood,
        x_bias_mu0, x_bias_sigma0, x_scale_hinges, x_scale_hinge_weights, sample_scales, use_point_estimates, sess)


"""
Run variational inference on splicing log-ratio linear regression.
"""
def estimate_splicing_linear_regression(
        init_feed_dict, splice_lr_loc, splice_lr_scale, x0_log, X_arr, sess=None):

    # don't need this since we already used the transcript expression
    # likelihood approximation to approximate splicing likelihood.
    init_feed_dict.clear()

    num_samples = splice_lr_loc.shape[0]
    num_features = splice_lr_scale.shape[1]

    splice_lr = tfd.Normal(
        loc=splice_lr_loc,
        scale=splice_lr_scale,
        name="splice_lr")

    make_likelihood = lambda qx: tf.reduce_sum(splice_lr.log_prob(qx))

    X = tf.constant(X_arr, dtype=tf.float32)

    # TODO: might try to find a better initialization
    x_init = np.zeros((num_samples, num_features), np.float32)

    w_mu0 = 0.0
    w_sigma0 = 10.0
    w_bias_mu0 = 0.0
    w_bias_sigma0 = 20.0

    return linear_regression_inference(
        init_feed_dict, X, x_init, make_likelihood,
        w_mu0, w_sigma0, w_bias_mu0, w_bias_sigma0)
