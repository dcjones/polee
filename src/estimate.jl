# TODO: use Pkg.dir when we make this an actual package
unshift!(PyVector(pyimport("sys")["path"]), "/home/dcjones/prj/extruder/src")
@pyimport tensorflow as tf
@pyimport tensorflow.python.client.timeline as tftl
@pyimport edward as ed
@pyimport edward.models as edmodels
@pyimport rnaseq_approx_likelihood
@pyimport hmc2

function load_samples(filenames)
    n = nothing

    musigma_tensors = []
    y0_tensors = []

    # work vectors for simplex!
    work1 = Array(Float32, 0)
    work2 = Array(Float32, 0)
    work3 = Array(Float32, 0)
    work4 = Array(Float32, 0)
    work5 = Array(Float32, 0)

    for filename in filenames
        input = h5open(filename, "r")

        n_ = read(input["n"])
        if n == nothing
            n = n_
            work1 = Array(Float32, n)
            work2 = Array(Float32, n)
            work3 = Array(Float32, n)
            work4 = Array(Float32, n)
            work5 = Array(Float32, n)
        elseif n != n_
            error("Prepare sample was run with different transcript annotations on some samples.")
        end

        μ = read(input["mu"])
        σ = read(input["sigma"])

        close(input)

        @assert length(μ) == n - 1
        @assert length(σ) == n - 1

        push!(musigma_tensors,
              tf.stack([tf.constant(PyVector(μ)), tf.constant(PyVector(σ))]))

        # choose mean to be the initial values for y
        initial_values = Array(Float32, n)
        simplex!(n, initial_values, work1, work2, work3, work4, work5, μ)
        map!(log, initial_values)
        push!(y0_tensors, initial_values)

    end

    musigma = tf.stack(musigma_tensors)
    y0 = tf.stack(y0_tensors)

    return (n, musigma, y0)
end


function estimate(experiment_spec_filename, output_filename)

    # read info from experiment specification
    experiment_spec = YAML.load_file(experiment_spec_filename)
    names = [entry["name"] for entry in experiment_spec]
    filenames = [entry["file"] for entry in experiment_spec]
    sample_factors = [get(entry, "factors", String[]) for entry in experiment_spec]
    num_samples = length(filenames)

    # build design matrix
    factoridx = Dict{String, Int}()
    factoridx["bias"] = 1
    for factors in sample_factors
        for factor in factors
            get!(factoridx, factor, length(factoridx) + 1)
        end
    end

    num_factors = length(factoridx)
    X_ = zeros(Float32, (num_samples, num_factors))
    for i in 1:num_samples
        for factor in sample_factors[i]
            j = factoridx[factor]
            X_[i, j] = 1
        end
    end
    X_[:, factoridx["bias"]] = 1
    X = tf.constant(X_)

    # read likelihood approximations and choose initial values
    n, musigma_data, y0 = load_samples(filenames)

    w_mu0 = 0.0
    #w_sigma0 = 0.1
    w_sigma0 = 0.001

    b_sigma_alpha0 = 0.1
    b_sigma_beta0 = 0.1

    # TODO
    # Here's after 2000 iterations
    # 7×4 DataFrames.DataFrame
    # │ Row │ factor  │ id     │ w       │ transcript_id     │
    # ├─────┼─────────┼────────┼─────────┼───────────────────┤
    # │ 1   │ "LAB_1" │ 184426 │ 2.78063 │ "ENST00000625598" │
    # │ 2   │ "LAB_2" │ 184426 │ 1.52607 │ "ENST00000625598" │
    # │ 3   │ "LAB_3" │ 184426 │ 1.57313 │ "ENST00000625598" │
    # │ 4   │ "LAB_4" │ 184426 │ 1.63165 │ "ENST00000625598" │
    # │ 5   │ "LAB_5" │ 184426 │ 1.58936 │ "ENST00000625598" │
    # │ 6   │ "LAB_7" │ 184426 │ 2.59445 │ "ENST00000625598" │
    # │ 7   │ "bias"  │ 184426 │ 2.77634 │ "ENST00000625598" │
    #
    # Why are these all positive? Wouldn't a more likely solution just bias
    # larger and some of the lab biases negative, or at least closer to zero?

    # IT IS MORE LIKELY CENTERED at ZERO!!! But the MAP optimization always goes
    # away from it. So either optimization is fucked, or there is some other
    # probability that does change somewhere somehow.
    #
    # So, things to try:
    # - See if musigma probability changes when I adjust w
    # - Try different optimization methods.
    #

    b_sigma = edmodels.InverseGamma(
            tf.constant(b_sigma_alpha0, shape=[1, n]),
            tf.constant(b_sigma_beta0, shape=[1, n]))

    B = edmodels.MultivariateNormalDiag(
            name="B",
            tf.zeros([num_samples, n]),
            tf.matmul(tf.ones([num_samples, 1]), b_sigma))

    W = edmodels.MultivariateNormalDiag(
            name="W",
            tf.constant(w_mu0, shape=[num_factors, n]),
            tf.concat(
                  [tf.constant(100.0, shape=[1, n]),
                   tf.constant(0.01, shape=[num_factors-1, n])], 0))
            #tf.constant(w_sigma0, shape=[num_factors, n]))
    #W = edmodels.Uniform(
            #tf.constant(-10.0, shape=[num_factors, n]),
            #tf.constant(10.0, shape=[num_factors, n]))

    #y = tf.add(tf.matmul(X, W), B)
    y = tf.matmul(X, W)

    musigma = rnaseq_approx_likelihood.RNASeqApproxLikelihood(
                y=y, value=musigma_data)

    datadict = PyDict(Dict(musigma => musigma_data))
    sess = ed.get_session()

    # Sampling
    # --------
    #=
    samples = tf.concat([tf.expand_dims(y0, 0),
                         tf.zeros([iterations, num_samples, n])], axis=0)
    qy_params = tf.Variable(samples, trainable=false)
    qy = edmodels.Empirical(params=qy_params)

    inference = hmc2.HMC2(PyDict(Dict(y => qy)),
                          data=PyDict(Dict(musigma => musigma_data)))
    inference[:run](step_size=0.00001, n_steps=2, logdir="logs")
    =#

    # VI
    # --

    # Optimize MAP for a few iterations to find a good starting point for VI
    qw_map_param = tf.Variable(tf.random_normal([num_factors, n]))
    qw_map = edmodels.PointMass(params=qw_map_param)

    qb_map_param = tf.Variable(tf.random_normal([num_samples, n]))
    qb_map = edmodels.PointMass(params=qb_map_param)

    #inference = ed.MAP(Dict(W => qw_map, B => qb_map), data=datadict)
    inference = ed.MAP(Dict(W => qw_map), data=datadict)

    inference[:run](n_iter=100)

    #optimizer = tf.train[:GradientDescentOptimizer](1e-8)
    #inference[:run](n_iter=200, optimizer=optimizer)

    qw_mu_ = sess[:run](qw_map_param)
    #@show qw_mu_[:,184426]
    #@show sess[:run](W[:log_prob](qw_map_param))
    lp0 = sum(sess[:run](W[:log_prob](qw_map_param)))
    musigma_ = rnaseq_approx_likelihood.RNASeqApproxLikelihood(
                y=tf.matmul(X, qw_map_param), value=musigma_data)
    ms_lp0 = sum(sess[:run](musigma_[:log_prob](musigma_data)))

    gs = tf.gradients(musigma_[:log_prob](musigma_data), [qw_map_param])
    @show qw_mu_[:,184426]
    @show sess[:run](gs[1])[:,184426]
    gs = tf.gradients(W[:log_prob](qw_map_param), [qw_map_param])
    @show sess[:run](gs[1])[:,184426]

    offset = minimum(qw_mu_[:,184426])
    qw_mu_[:,184426] -= offset
    qw_mu_[factoridx["bias"],184426] += 2 * offset
    qw_map_param_ = tf.constant(qw_mu_)
    #@show qw_mu_[:,184426]
    #@show sess[:run](W[:log_prob](tf.constant(qw_mu_)))
    lp1 = sum(sess[:run](W[:log_prob](qw_map_param_)))
    musigma_ = rnaseq_approx_likelihood.RNASeqApproxLikelihood(
                y=tf.matmul(X, qw_map_param_), value=musigma_data)
    ms_lp1 = sum(sess[:run](musigma_[:log_prob](musigma_data)))

    gs = tf.gradients(musigma_[:log_prob](musigma_data), [qw_map_param_])
    @show sess[:run](gs[1])[:,184426]
    gs = tf.gradients(W[:log_prob](qw_map_param_), [qw_map_param_])
    @show sess[:run](gs[1])[:,184426]


    @show (lp0, lp1, lp0 < lp1)
    @show (ms_lp0, ms_lp1, ms_lp0 < ms_lp1)
    @show (-(lp0 + ms_lp0), -(lp1 + ms_lp1), lp0 + ms_lp0 < lp1 + ms_lp1)

    # TODO: It seems like the gradients for y wrt to qw_map_param will always
    # push it above the actual maximum posterior. How could this be? I think
    # there is something to do with the overparameterization that we're still
    # missing.
    #
    # Is it possible that there are multiple modes and it's getting stuck in
    # one? Not in this case because bias + ε, w - ε will lead to strictly
    # improved, so there should be a smooth upward gradient, it's just negative
    # in any single direction.

    # I'm completely at a loss.

    iterations = 200

    qw_mu = tf.Variable(sess[:run](qw_map_param))
    qw_sigma = tf.identity(tf.Variable(tf.fill([num_factors, n], 0.1)))
    qw = edmodels.MultivariateNormalDiag(name="qw", qw_mu, qw_sigma)

    qb_mu = tf.Variable(sess[:run](qb_map_param))
    qb_sigma = tf.identity(tf.Variable(tf.fill([num_samples, n], 0.1)))
    qb = edmodels.MultivariateNormalDiag(name="qb", qb_mu, qb_sigma)

    inference = ed.KLqp(Dict(W => qw, B => qb), data=datadict)

    # TODO: experiment with making this more aggressive
    learning_rate = 1e-2
    beta1 = 0.7
    beta2 = 0.99
    optimizer = tf.train[:AdamOptimizer](learning_rate, beta1, beta2)
    inference[:run](n_iter=iterations, optimizer=optimizer)

    # Trace
    #run_options = tf.RunOptions(trace_level=tf.RunOptions[:FULL_TRACE])
    #run_metadata = tf.RunMetadata()
    #sess[:run](inference[:train], options=run_options,
               #run_metadata=run_metadata)
    #sess[:run](inference[:loss], options=run_options,
               #run_metadata=run_metadata)

    #tl = tftl.Timeline(run_metadata[:step_stats])
    #ctf = tl[:generate_chrome_trace_format]()
    #trace_out = pybuiltin(:open)("timeline.json", "w")
    #trace_out[:write](ctf)
    #trace_out[:close]()

    # MAP
    # ---
    #=
    qy_params = tf.Variable(y0)
    qy = edmodels.PointMass(params=qy_params)
    #qy = edmodels.PointMass(params=tf.Variable(tf.zeros([num_samples, n])))
    inference = ed.MAP(Dict(y => qy), data=datadict)
    inference[:run](n_iter=iterations)
    =#

    # Evaluate posterior means
    #y = tf.add(tf.matmul(X, qw_mu), qb_mu)
    y = tf.matmul(X, qw_mu)

    post_mean = sess[:run](tf.nn[:softmax](y, dim=-1))
    #post_mean = sess[:run](y)
    write_estimates("post_mean.csv", names, post_mean)

    # E-GEUV specific code: remove batch effects by zeroing out that part of the
    # design matrix.
    batch_effect_factor_pattern = r"LAB_"
    for (factor, idx) in factoridx
        if match(batch_effect_factor_pattern, factor) != nothing
            X_[:, idx] = 0
        end
    end
    Z = tf.constant(X_)
    #y = tf.add(tf.matmul(Z, qw_mu), qb_mu)
    y = tf.matmul(Z, qw_mu)

    @show sess[:run](qb_mu)[184426]

    post_mean = sess[:run](tf.nn[:softmax](y, dim=-1))
    #post_mean = sess[:run](y)
    write_estimates("post_mean_nobatch.csv", names, post_mean)

    write_effects("effects.csv", factoridx, sess[:run](qw_mu))
end


function write_effects(filename, factoridx, W)
    n = size(W, 2)
    open(filename, "w") do output
        println(output, "factor,id,w")
        for factor = sort(collect(keys(factoridx)))
            idx = factoridx[factor]
            for j in 1:n
                @printf(output, "%s,%d,%e\n", factor, j, W[idx, j])
            end
        end
    end
end


function write_estimates(filename, names, est)
    n = size(est, 2)
    open(filename, "w") do output
        # TODO: where do we get the transcript names from?
        println(output, "name,id,tpm")
        for (i, name) in enumerate(names)
            for j in 1:n
                @printf(output, "%s,%d,%e\n", name, j, est[i, j])
            end
        end
    end
end
