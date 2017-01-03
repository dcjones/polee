
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

    # step size constants
    ss_τ = 1.0
    ss_ε = 1e-16
    ss_α = 0.1
    ss_η = 1.0
    ss_max_μ_step = 5e-1
    ss_max_ω_step = 1e-1
    srand(1234)

    # number of monte carlo samples to estimate gradients an elbo at each
    # iteration
    num_mc_samples = 1
    η = fillpadded(FloatVec, 0.0f0, n)
    ζ = fillpadded(FloatVec, 0.0f0, n)

    μ = fillpadded(FloatVec, 0.0f0, n)
    σ = fillpadded(FloatVec, 1.0f0, n, 1.f0)
    ω = fillpadded(FloatVec, 0.0f0, n) # log transformed σ

    π_grad = fillpadded(FloatVec, 0.0f0, n)
    μ_grad = fillpadded(FloatVec, 0.0f0, n)
    ω_grad = fillpadded(FloatVec, 0.0f0, n)

    # step-size
    s_μ = fillpadded(FloatVec, 1e-6, n)
    s_ω = fillpadded(FloatVec, 1e-6, n)

    # simd vectors
    ηv = reinterpret(FloatVec, η)
    ζv = reinterpret(FloatVec, ζ)
    μv = reinterpret(FloatVec, μ)
    ωv = reinterpret(FloatVec, ω)
    σv = reinterpret(FloatVec, σ)
    s_μv = reinterpret(FloatVec, s_μ)
    s_ωv = reinterpret(FloatVec, s_ω)
    μ_gradv = reinterpret(FloatVec, μ_grad)
    ω_gradv = reinterpret(FloatVec, ω_grad)

    elbo = 0.0
    elbo0 = 0.0
    max_elbo = -Inf # smallest elbo seen so far

    step_num = 0
    small_step_count = 0
    fruitless_step_count = 0

    # stopping criteria
    max_small_steps = 2
    max_fruitless_steps = 20
    max_steps = 200

    while true
        step_num += 1
        elbo0 = elbo
        elbo = 0.0
        fill!(μ_grad, 0.0f0)
        fill!(ω_grad, 0.0f0)
        map!(exp, σv, ωv)

        for _ in 1:num_mc_samples
            rand!(Normal(), η)

            # de-standardize normal variate
            for i in 1:length(ζv)
                ζv[i] = σv[i] .* ηv[i] + μv[i]
            end

            elbo += log_likelihood(model, mklX, ζ, π_grad)
            μ_grad .+= π_grad
            ω_grad .+= π_grad .* η .* σ
        end
        μ_grad /= num_mc_samples
        ω_grad /= num_mc_samples
        ω_grad .+= 1
        elbo /= num_mc_samples
        elbo += normal_entropy!(ω_grad, σ)
        max_elbo = max(max_elbo, elbo)
        @show elbo

        if step_num == 1
            s_μ[:] = μ_grad.^2
            s_ω[:] = ω_grad.^2
        end

        c = ss_η * step_num^(-0.5 + ss_ε)

        for i in 1:length(μ)
            ρ = c / (ss_τ + sqrt(s_μ[i]))
            μ[i] += clamp(ρ * μ_grad[i], -ss_max_μ_step, ss_max_μ_step)
            s_μ[i] = (1 - ss_α) * s_μ[i] + ss_α * μ_grad[i]^2
        end

        for i in 1:length(ω)
            ρ = c / (ss_τ + sqrt(s_ω[i]))
            ω[i] += clamp(ρ * ω_grad[i], -ss_max_ω_step, ss_max_ω_step)
            s_ω[i] = (1 - ss_α) * s_ω[i] + ss_α * ω_grad[i]^2
        end

        if elbo < max_elbo
            fruitless_step_count += 1
        else
            fruitless_step_count = 0
        end

        if abs((elbo - elbo0) / elbo) < 1e-4
            small_step_count += 1
        else
            small_step_count = 0
        end

        if small_step_count > max_small_steps ||
           fruitless_step_count > max_fruitless_steps ||
           step_num > max_steps
            break
        end
    end

    @show step_num
end


main()


