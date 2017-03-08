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
        push!(y0_tensors, initial_values)

    end

    musigma = tf.stack(musigma_tensors)
    y0 = tf.stack(y0_tensors)

    return (n, musigma, y0)
end


function estimate(experiment_spec_filename, output_filename)

    experiment_spec = YAML.load_file(experiment_spec_filename)
    filenames = [entry["filename"] for entry in experiment_spec]
    num_samples = length(filenames)

    n, musigma_data, y0 = load_samples(filenames)

    scale_mu0 = 0.0
    scale_sigma0 = 1.5

    # TODO: actually interesting model
    mu0 = 0.0
    sigma0 = 10.0
    y = edmodels.MultivariateNormalDiag(tf.constant(mu0, shape=[num_samples, n]),
                                        tf.constant(sigma0, shape=[num_samples, n]))

    musigma = rnaseq_approx_likelihood.RNASeqApproxLikelihood(
                y=y, scale_mu0=scale_mu0, scale_sigma0=scale_sigma0,
                value=musigma_data)

    iterations = 1000

    # Sampling
    #=
    samples = tf.concat([tf.expand_dims(y0, 0),
                         tf.zeros([iterations, num_samples, n])], axis=0)
    qy = edmodels.Empirical(params=tf.Variable(samples, trainable=false))

    inference = hmc2.HMC2(PyDict(Dict(y => qy)),
                          data=PyDict(Dict(musigma => musigma_data)))
    inference[:run](step_size=0.00001, n_steps=2, logdir="logs")
    =#

    # VI
    qy_mu = tf.Variable(tf.random_normal([num_samples, n]))
    qy_sigma = tf.nn[:softplus](tf.Variable(tf.random_normal([num_samples, n])))
    qy = edmodels.MultivariateNormalDiag(qy_mu, qy_sigma)
    inference = ed.KLqp(Dict(y=> qy), data=PyDict(Dict(musigma => musigma_data)))
    inference[:run](n_iter=iterations)
end

