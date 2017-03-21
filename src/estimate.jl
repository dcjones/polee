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
    @show X_
    X = tf.constant(X_)

    # read likelihood approximations and choose initial values
    n, musigma_data, y0 = load_samples(filenames)

    w_mu0 = 0.0
    w_sigma0 = 1.0

    b_mu0 = log(1/n)
    b_sigma0 = 10.0

    W = edmodels.MultivariateNormalDiag(
            tf.constant(w_mu0, shape=[num_factors, n]),
            tf.constant(w_sigma0, shape=[num_factors, n]))

    B = edmodels.MultivariateNormalDiag(
            tf.constant(b_mu0, shape=[num_samples, n]),
            tf.constant(b_sigma0, shape=[num_samples, n]))

    y = tf.add(tf.matmul(X, W), B)

    musigma = rnaseq_approx_likelihood.RNASeqApproxLikelihood(
                y=y, value=musigma_data)

    iterations = 5000
    #iterations = 50
    datadict = PyDict(Dict(musigma => musigma_data))
    #sess_config = tf.ConfigProto(intra_op_parallelism_threads=16,
                                 #inter_op_parallelism_threads=16)
    #sess = tf.InteractiveSession(config=sess_config)
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
    qw_mu = tf.Variable(tf.zeros([num_factors, n]))
    qw_sigma = tf.identity(tf.Variable(tf.fill([num_factors, n], 1.0)))
    qw = edmodels.MultivariateNormalDiag(name="qw", qw_mu, qw_sigma)

    qb_mu = tf.Variable(y0)
    qb_sigma = tf.identity(tf.Variable(tf.fill([num_samples, n], 1.0)))
    qb = edmodels.MultivariateNormalDiag(name="qb", qb_mu, qb_sigma)

    inference = ed.KLqp(Dict(W => qw, B => qb), data=datadict)
    #inference = ed.KLqp(Dict(B => qb), data=datadict)
    #inference[:run](n_iter=iterations)

    inference[:run](n_iter=iterations,
                    optimizer=tf.train[:MomentumOptimizer](1e-7, 0.6))

    # Trace
    #=
    run_options = tf.RunOptions(trace_level=tf.RunOptions[:FULL_TRACE])
    run_metadata = tf.RunMetadata()
    sess[:run](inference[:train], options=run_options,
               run_metadata=run_metadata)
    #sess[:run](inference[:loss], options=run_options,
               #run_metadata=run_metadata)

    tl = tftl.Timeline(run_metadata[:step_stats])
    ctf = tl[:generate_chrome_trace_format]()
    trace_out = pybuiltin(:open)("timeline.json", "w")
    trace_out[:write](ctf)
    trace_out[:close]()
    =#

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

    @show sess[:run](tf.reduce_min(qw_mu))
    @show sess[:run](tf.reduce_max(qw_mu))

    y = tf.add(tf.matmul(X, qw_mu), qb_mu)
    post_mean = sess[:run](tf.nn[:softmax](y, dim=-1))
    #post_mean = sess[:run](tf.nn[:softmax](qy_mu, dim=-1))

    # TODO: some way of computing values after subtracting out particular
    # effects

    open("post_mean.csv", "w") do output
        # TODO: where do we get the transcript names from?
        println(output, "name,id,tpm")
        for (i, name) in enumerate(names)
            for j in 1:n
                println(output, name, ",", j, ",", 1e6 * post_mean[i, j])
            end
        end
    end
end

