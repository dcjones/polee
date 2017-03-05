

function load_sample(filename)
    # TODO:
    # Read yaml file.
    # Generate a MultivariateNormalDiag models along with metadata used for
    # model design.
end


function estimate(input_filename, output_filename)
    # TODO: use Pkg.dir when we make this an actual package
    unshift!(PyVector(pyimport("sys")["path"]), "/home/dcjones/prj/extruder/src")
    @pyimport tensorflow as tf
    @pyimport edward as ed
    @pyimport edward.models as edmodels
    @pyimport rnaseq_approx_likelihood

    input = h5open(input_filename, "r")
    scale_σ = 1.0 # TODO: I honestly have no idea what this should be
    n = read(input["n"])
    μ = read(input["mu"])
    σ = read(input["sigma"])
    @assert length(μ) == n - 1
    @assert length(σ) == n - 1

    @show minimum(μ)
    @show maximum(μ)
    @show μ[1]
    @show μ[end]

    pi = rnaseq_approx_likelihood.RNASeqApproxLikelihood(
            mu=tf.constant(PyVector(μ)),
            sigma=tf.constant(PyVector(σ)),
            scale_sigma=scale_σ,
            value=tf.zeros([n]))

    # choose initial values
    initial_values = zeros(Float32, n)
    work1 = Array(Float32, n)
    work2 = Array(Float32, n)
    work3 = Array(Float32, n)
    work4 = Array(Float32, n)
    work5 = Array(Float32, n)

    simplex!(n, initial_values, work1, work2, work3, work4, work5, μ)
    map!(log, initial_values)

    num_samples = 50
    samples = tf.concat([tf.expand_dims(initial_values, 0),
                         tf.zeros([num_samples, n])], axis=0)
    qpi = edmodels.Empirical(params=tf.Variable(samples))

    inference = ed.HMC(PyDict(Dict(pi => qpi)))
    inference[:run](step_size=0.0001)
    #inference[:run](step_size=0.000001)
end

