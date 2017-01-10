
using HDF5

include("fastmath.jl")
include("mkl.jl")
include("model.jl")


function main()
    input = h5open("output.h5", "r")
    m = read(input["m"]) # number of reads
    n = read(input["n"]) # number of transcripts
    colptr = read(input["colptr"])
    rowval = read(input["rowval"])
    nzval = read(input["nzval"])
    @show (m, n)

    srand(4322)

    X = SparseMatrixCSC(m, n, colptr, rowval, nzval)
    mklX = MKLSparseMatrixCSC(X)
    π0 = zeros(Float32, n) # mixture coefficients
    π = zeros(Float32, n) # proposal
    ϕ0 = zeros(Float32, n) # momentum
    ϕ = zeros(Float32, n) # momentum
    grad0 = zeros(Float32, n) # gradient wrt π
    grad = zeros(Float32, n) # gradient wrt π
    ϵ = 1.0
    σ_ϕ = 1e-1

    model = Model(m, n)

    # index samples to output
    output_idx = 99000
    #output_idx = 99
    out = open("hmc-samples.csv", "w")

    burnin = 20
    #num_samples = 2200
    num_samples = 120
    num_sub_steps = 10
    lp0 = log_likelihood(model, mklX, π, grad0)
    @show lp0
    accept_count = 0

    # step size heuritic
    ss_μ = log(1e-2)
    ss_γ = 0.05
    ss_κ = 0.75
    ss_t0 = 20
    ss_hsum = 0.0

    ss_logϵ = ss_μ
    ss_logϵ_mean = ss_logϵ

    #max_delta = 5e-1
    max_delta = 1e-1
    #zero_count = 3*div(n,4)

    for sample_num in 1:num_samples
        println()
        @show sample_num
        #π0, π = π, π0
        #ϕ0, ϕ = ϕ, ϕ0

        #rand!(ϕ_dist, ϕ0)
        randn!(ϕ0)
        ϕ0 .*= σ_ϕ
        #ϕ0[:] = 0.0 # just optimize please

        copy!(ϕ, ϕ0)
        copy!(π, π0)
        copy!(grad, grad0)
        lp = -Inf32
        #ϵ = max(1e-3, exp(ss_logϵ_mean))
        ϵ = max(1e-6, exp(ss_logϵ_mean))
        @show ϵ

        # Version 1
        #for step_num in 1:num_sub_steps
            #ϕ .+= clamp(0.5 * ϵ * grad, -1e-2, 1e-2)
            #π .+= ϵ * ϕ
            #lp = log_likelihood(model, mklX, π, grad)
            #ϕ .+= clamp(0.5 * ϵ * grad, -1e-2, 1e-2)
        #end

        # Version 2
        ϕ .+= clamp(0.5 * ϵ * grad, -max_delta, max_delta)
        @show π0[output_idx]
        for step_num in 1:num_sub_steps
            π .+= ϵ * ϕ
            lp = log_likelihood(model, mklX, π, grad)
            @show ϕ[output_idx]
            @show grad[output_idx]
            @show π[output_idx]
            ϕ .+= clamp(ϵ * grad, -max_delta, max_delta)
        end
        ϕ .-= clamp(0.5 * ϵ * grad, -max_delta, max_delta)
        ϕ *= -1.0

        #r = (lp + logpdf(ϕ_dist, ϕ)) - (lp0 + logpdf(ϕ_dist, ϕ0))
        r = (lp - 0.5 * sumabs2(σ_ϕ * ϕ)) - (lp0 - 0.5 * sumabs2(σ_ϕ * ϕ0))
        if !isfinite(r)
            r = -Inf
        end

        # adapt epsilon parameter
        #if sample_num < 30
            ss_hsum += 0.65 - min(1.0, exp(r))
            ss_logϵ = ss_μ - ss_hsum * (sqrt(sample_num) /
                                          (ss_γ * (sample_num + ss_t0)))
            ss_η = sample_num^-ss_κ
            ss_logϵ_mean = ss_η * ss_logϵ + (1 - ss_η) * ss_logϵ_mean
        #end

        #@show 0.5 * dot(ϕ, ϕ)
        #@show 0.5 * dot(ϕ0, ϕ0)
        #@show lp
        #@show lp0

        #@show lp0
        @show lp
        #@show lp - lp0
        #@show sum(π0 .- π)
        #@show all(isfinite(π))

        #@show dot(ϕ, ϕ) - dot(ϕ0, ϕ0)
        #@show dot(ϕ, ϕ)
        #a = indmax(abs(π0 .- π))
        #@show (a, π[a], π0[a], grad[a], ϕ[a])

        #@show logpdf(ϕ_dist, ϕ)
        #@show logpdf(ϕ_dist, ϕ0)

        @show r
        #@show min(1.0, exp(r))

        #if sum(π0 .- π) < 1e-10 && r < 0.0
            #@show log_likelihood(model, mklX, π0, grad)
            #@show log_likelihood(model, mklX, π, grad)
            #@show maximum(abs(π0 .- π))

            #error("WTF")
        #end

        #@show (lp, lp0, logpdf(ϕ_dist, ϕ), logpdf(ϕ_dist, ϕ0))
        # accept proposal
        if log(rand()) < r
            accept_count += 1
            copy!(π0, π)
            copy!(grad0, grad)
            lp0 = lp
            println("ACCEPT")
        end

        if sample_num > burnin
            log_likelihood(model, mklX, π, grad)
            @printf(out, "%.12f\n", model.π_simplex[output_idx])
        end

        #@show π[output_idx]
        @show grad[output_idx]
        @show model.π_simplex[output_idx]

        #@show lp
    end

    close(out)

    @show accept_count / num_samples

end


main()
