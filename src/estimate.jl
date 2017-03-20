# TODO: use Pkg.dir when we make this an actual package
unshift!(PyVector(pyimport("sys")["path"]), "/home/dcjones/prj/extruder/src")
@pyimport tensorflow as tf
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
        @show initial_values[1:10]
        push!(y0_tensors, initial_values)

    end

    musigma = tf.stack(musigma_tensors)
    y0 = tf.stack(y0_tensors)

    return (n, musigma, y0)
end


function estimate(experiment_spec_filename, output_filename)

    experiment_spec = YAML.load_file(experiment_spec_filename)
    names = [entry["name"] for entry in experiment_spec]
    filenames = [entry["file"] for entry in experiment_spec]
    num_samples = length(filenames)

    n, musigma_data, y0 = load_samples(filenames)
    @show n

    # I really need to think of some principled way of setting this
    scale_mu0 = 0.0
    scale_sigma0 = 10.0

    mu0 = log(1/n)
    @show mu0
    sigma0 = 10.0
    y = edmodels.MultivariateNormalDiag(tf.constant(mu0, shape=[num_samples, n]),
                                        tf.constant(sigma0, shape=[num_samples, n]))

    musigma = rnaseq_approx_likelihood.RNASeqApproxLikelihood(
                y=y, scale_mu0=scale_mu0, scale_sigma0=scale_sigma0,
                value=musigma_data)

    iterations = 10000
    #iterations = 50

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

    datadict = PyDict(Dict(musigma => musigma_data))

    # VI
    # --
    #qy_mu = tf.Variable(tf.random_normal([num_samples, n]))
    qy_mu = tf.Variable(y0)
    qy_sigma = tf.nn[:softplus](tf.Variable(tf.random_normal([num_samples, n])))
    qy = edmodels.MultivariateNormalDiag(qy_mu, qy_sigma)
    inference = ed.KLqp(Dict(y => qy), data=datadict)
    #inference[:run](n_iter=iterations,
                    #optimizer=tf.train[:GradientDescentOptimizer](1e-7))
    inference[:run](n_iter=iterations,
                    optimizer=tf.train[:MomentumOptimizer](1e-7, 0.6))
    #inference[:run](n_iter=iterations,
                    #optimizer=tf.train[:AdagradOptimizer](1e-7))
    #inference[:run](n_iter=iterations,
                    #optimizer=tf.train[:AdamOptimizer](1e-5))

    # 5990485

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
    sess = ed.get_session()
    #post_mean = sess[:run](tf.nn[:softmax](qy_mu, dim=-1))

    post_mean = sess[:run](tf.nn[:softmax](qy, dim=-1))

    #@show sess[:run](qy_params)[iterations,1:10]
    #@show sess[:run](qy_params)[iterations,end-10:end]
    #@show sess[:run](tf.reduce_min(qy_params))
    #@show sess[:run](tf.reduce_max(qy_params))

    #@show sess[:run](qy_params)[1:10]
    #@show sess[:run](qy_params)[end-10:end]

    @show sess[:run](musigma_data)[1,1:10]
    @show sess[:run](qy_mu)[1:10]
    @show sess[:run](qy_mu)[end-10:end]
    #@show sess[:run](tf.reduce_min(qy_sigma))
    #@show sess[:run](tf.reduce_max(qy_sigma))
    @show sess[:run](qy_sigma)[1:10]
    @show sess[:run](qy_sigma)[end-10:end]

    #@show sess[:run](tf.reduce_sum(tf.exp(qy_mu)))
    @show sess[:run](tf.reduce_min(tf.exp(qy_mu)))
    @show sess[:run](tf.reduce_max(tf.exp(qy_mu)))
    #@show sess[:run](tf.argmin(tf.exp(qy_mu), axis=-1))
    #@show sess[:run](tf.argmax(tf.exp(qy_mu), axis=-1))
    #@show sess[:run](tf.log(tf.divide(1.0, tf.to_float(tf.range(n - 1, 0, -1)))))[1:10]
    #@show sess[:run](tf.log(tf.divide(1.0, tf.to_float(tf.range(n - 1, 0, -1)))))[end-10:end]

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

