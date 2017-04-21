
include("randn.jl")

function approximate_likelihood(input_filename::String, output_filename::String)
    sample = read(input_filename, RNASeqSample)
    approximate_likelihood(sample, output_filename)
end


function approximate_likelihood(sample::RNASeqSample, output_filename::String)
    μ, σ = approximate_likelihood(sample)
    h5open(output_filename, "w") do out
        n = sample.n
        out["n"] = sample.n
        out["mu", "compress", 1] = μ[1:n-1]
        out["sigma", "compress", 1] = σ[1:n-1]
        g = g_create(out, "metadata")
        attrs(g)["gfffilename"] = sample.transcript_metadata.filename
        attrs(g)["gffhash"]     = base64encode(sample.transcript_metadata.gffhash)
        attrs(g)["gffsize"]     = sample.transcript_metadata.gffsize
    end
end

function approximate_likelihood_from_isolator(input_filename,
                                              effective_lengths_filename,
                                              output_filename)
    input = open(input_filename)

    n = parse(Int, readline(input))

    m = parse(Int, readline(input))
    @show (m, n)
    I = Array(Int, 0)
    J = Array(Int, 0)
    V = Array(Float32, 0)
    println("reading isolator likelihood matrix...")
    for line in eachline(input)
        j_, i_, v_ = split(line, ',')
        i = 1 + parse(Int, i_)
        j = 1 + parse(Int, j_)
        v = parse(Float32, v_)

        push!(I, i)
        push!(J, j)
        push!(V, v)
    end
    X = sparse(I, J, V, m, n)
    rsbX = SparseMatrixRSB(X)
    println("done")

    println("reading isolator transcript weights...")
    effective_lengths = Float32[]
    open(effective_lengths_filename) do input
        for line in eachline(input)
            push!(effective_lengths, parse(Float32, line))
        end
    end
    println("done")

    sample = RNASeqSample(m, n, rsbX, effective_lengths, TranscriptsMetadata())

    # measure and dump connectivity info
    #=
    p = sortperm(I)
    I = I[p]
    J = J[p]
    v = V[p]
    edge_count = Dict{Tuple{Int, Int}, Int}()
    a = 1
    while a <= length(I)
        if I[a] % 1000 == 0
            @show I[a]
        end

        b = a
        while b <= length(I) && I[a] == I[b]
            b += 1
        end

        for k in a:b-1
            for l in k+1:b-1
                key = J[k] < J[l] ? (J[k], J[l]) : (J[l], J[k])
                if haskey(edge_count, key)
                    edge_count[key] += 1
                else
                    edge_count[key] = 1
                end
            end
        end
        a = b
    end

    open("connectivity.csv", "w") do out
        for ((u,v), c) in edge_count
            println(out, u, ",", v, ",", c)
        end
    end
    =#

    #@profile μ, σ = approximate_likelihood(sample)
    #Profile.print()

    @time μ, σ = approximate_likelihood(sample)

    h5open(output_filename, "w") do out
        n = sample.n
        out["n"] = sample.n
        out["mu", "compress", 1] = μ[1:n-1]
        out["sigma", "compress", 1] = σ[1:n-1]
    end
end


"""
Compute the entropy of a multivariate normal distribution along with gradients
wrt σ.

Note: when σ² does not have length divisible by the simd vector length it needs
to be padded with extra 1s
"""
function normal_entropy!(grad, σ, n)
    vs = reinterpret(FloatVec, σ)
    vs_sumlogs = sum(mapreduce(log, +, zero(FloatVec), vs))
    entropy = 0.5 * n * log(2 * π * e) + vs_sumlogs

    return entropy
end


"""
Sample from the variational distribution and evaluate the true likelihood
function at these samples.
"""
function diagnostic_samples(model, X, μ, σ, i)
    n = length(μ)

    num_samples = 500
    samples = Array(Float32, num_samples)
    weights = Array(Float32, num_samples)
    π = Array(Float32, n)
    π_grad = Array(Float32, n)

    for k in 1:num_samples
        for j in 1:n
            π[j] = rand(Normal(μ[j], σ[j]))
        end

        ll = log_likelihood(model, X, π, π_grad)
        samples[k] = model.π_simplex[i]
        weights[k] = ll
    end

    return (samples, weights)
end


function myrandn!(xs::Vector{Float32})
    for i in 1:length(xs)
        xs[i] = randn()
    end
end


function approximate_likelihood(s::RNASeqSample)
    m, n = size(s)
    model = Model(m, n)

    # step size constants
    ss_τ = 1.0
    ss_ε = 1e-16

    # influence of the most recent gradient on step size
    ss_ω_α = 0.1
    ss_μ_α = 0.01

    ss_η = 1.0

    ss_max_μ_step = 1e-1
    ss_max_ω_step = 1e-1
    srand(4324)

    # number of monte carlo samples to estimate gradients an elbo at each
    # iteration
    num_mc_samples = 2
    η = fillpadded(FloatVec, 0.0f0, n-1)
    ζ = fillpadded(FloatVec, 0.0f0, n-1)

    μ = fillpadded(FloatVec, 0.0f0, n-1)
    σ = fillpadded(FloatVec, 1.0f0, n-1, 1.f0)
    ω = fillpadded(FloatVec, -3.0f0, n-1) # log transformed σ

    π_grad = fillpadded(FloatVec, 0.0f0, n-1)
    μ_grad = fillpadded(FloatVec, 0.0f0, n-1)
    ω_grad = fillpadded(FloatVec, 0.0f0, n-1)

    # step-size
    s_μ = fillpadded(FloatVec, 1e-6, n-1)
    s_ω = fillpadded(FloatVec, 1e-6, n-1)

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

    output_idx = 72281

    println("Optimizing ELBO: ", -Inf)

    while true
        step_num += 1
        elbo0 = elbo
        elbo = 0.0
        fill!(μ_grad, 0.0f0)
        fill!(ω_grad, 0.0f0)
        map!(exp, σv, ωv)

        for _ in 1:num_mc_samples
            for i in 1:n-1
                η[i] = _randn()
            end

            # de-standardize normal variate
            for i in 1:length(ζv)
                ζv[i] = σv[i] .* ηv[i] + μv[i]
            end

            lp = log_likelihood(model, s.X, s.effective_lengths, ζ, π_grad)
            @assert isfinite(lp)
            elbo += lp

            @inbounds for i in 1:n-1
                μ_grad[i] += π_grad[i]
                ω_grad[i] += π_grad[i] * η[i] * σ[i]
            end
        end

        for i in 1:n-1
            μ_grad[i] /= num_mc_samples
            ω_grad[i] /= num_mc_samples
            ω_grad[i] += 1 # normal distribution entropy gradient
        end

        elbo /= num_mc_samples
        elbo += normal_entropy!(ω_grad, σ, n-1)::Float64
        max_elbo = max(max_elbo, elbo)
        @assert isfinite(elbo)
        @printf("\e[F\e[JOptimizing ELBO: %.4e\n", elbo)
        #@printf("Optimizing ELBO: %.4e\n", elbo)

        if step_num == 1
            s_μ[:] = μ_grad.^2
            s_ω[:] = ω_grad.^2
        end

        c = ss_η * (step_num^(-0.5 + ss_ε))::Float64
        for i in 1:n-1
            s_μ[i] = (1 - ss_μ_α) * s_μ[i] + ss_μ_α * μ_grad[i]^2
            ρ = c / (ss_τ + sqrt(s_μ[i]))
            μ[i] += clamp(ρ * μ_grad[i], -ss_max_μ_step, ss_max_μ_step)
        end

        for i in 1:n-1
            s_ω[i] = (1 - ss_ω_α) * s_ω[i] + ss_ω_α * ω_grad[i]^2
            ρ = c / (ss_τ + sqrt(s_ω[i]))
            ω[i] += clamp(ρ * ω_grad[i], -ss_max_ω_step, ss_max_ω_step)
        end

        #if step_num > 600
            #break
        #end

        if elbo < max_elbo
            fruitless_step_count += 1
        else
            fruitless_step_count = 0
        end

        if fruitless_step_count > 5
            break
        end
    end

    println("Finished in ", step_num, " steps")

    # Write out point estimates for convenience
    #log_likelihood(model, s.X, s.effective_lengths, μ, π_grad)
    #open("point-estimates.csv", "w") do out
        #for i in 1:n
            #@printf(out, "%e\n", model.π_simplex[i])
        #end
    #end

    map!(exp, σv, ωv)
    return μ, σ
end


