
using Distributions
using HDF5

include("fastmath.jl")
include("mkl.jl")
include("model.jl")

using FastMath


"""
Compute the entropy of a multivariate normal distribution along with gradients
wrt σ.

Note: when σ² does not have length divisible by the simd vector length it needs
to be padded with extra 1s
"""
function normal_entropy!(grad, σ)
    n = length(σ)
    vs = reinterpret(FloatVec, σ)
    vs_sumlogs = sum(mapreduce(log, +, zero(FloatVec), vs))
    entropy = n * log(2 * π * e) + vs_sumlogs

    gradv = reinterpret(FloatVec, grad)
    for i in 1:length(vs)
        gradv[i] += inv(vs[i])
    end

    @show entropy

    return entropy
end


function main()
    input = h5open(ARGS[1], "r")
    m = read(input["m"]) # number of reads
    n = read(input["n"]) # number of transcripts
    colptr = read(input["colptr"])
    rowval = read(input["rowval"])
    nzval = read(input["nzval"])

    X = SparseMatrixCSC(m, n, colptr, rowval, nzval)
    mklX = MKLSparseMatrixCSC(X)

    model = Model(m, n)

    # number of monte carlo samples to estimate gradients an elbo at each
    # iteration
    num_mc_samples = 5
    η = fillpadded(FloatVec, 0.0f0, n)
    ζ = fillpadded(FloatVec, 0.0f0, n)

    μ = fillpadded(FloatVec, 0.0f0, n)
    σ = fillpadded(FloatVec, 1.0f0, n, 1.f0)
    ω = fillpadded(FloatVec, 0.0f0, n) # log transformed σ

    π_grad = fillpadded(FloatVec, 0.0f0, n)
    μ_grad = fillpadded(FloatVec, 0.0f0, n)
    ω_grad = fillpadded(FloatVec, 0.0f0, n)

    # simd vectors
    ηv = reinterpret(FloatVec, η)
    ζv = reinterpret(FloatVec, ζ)
    μv = reinterpret(FloatVec, μ)
    ωv = reinterpret(FloatVec, ω)
    σv = reinterpret(FloatVec, σ)

    elbo = 0.0
    elbo0 = 0.0

    step_num = 0
    while true
        step_num += 1
        elbo0 = elbo
        elbo = 0.0
        fill!(μ_grad, 0.0f0)
        fill!(ω_grad, 1.0f0)
        map!(exp, σv, ωv)

        for _ in 1:num_mc_samples
            rand!(Normal(), η)

            # de-standardize normal variate
            for i in 1:length(ζv)
                ζv[i] = σv[i] .* ηv[i] + μv[i]

                # TODO: test directly optimizing π
                #ζv[i] = μv[i]
            end

            elbo += log_likelihood(model, mklX, ζ, π_grad)
            μ_grad += π_grad
            ω_grad += π_grad .* η .* σ
        end
        μ_grad /= num_mc_samples
        elbo /= num_mc_samples
        elbo += normal_entropy!(ω_grad, σ)

        # TODO: This seems wrong. Where do I take into account the σ = exp(ω)
        # TODO: what do I do with the gradient for entropy???

        println("-------------------")
        @show elbo
        @show μ[1:5]
        @show μv[1]

        μ .+= 1e-6 * μ_grad
        ω .+= 1e-8 * ω_grad

        # TODO: gradient descent step

        # TODO: termination condition

        #break # XXX

        if step_num > 20
            break
        end
    end
end


main()


