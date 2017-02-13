

function approximate_likelihood(input_filename, output_filename)
    sample = read(input_filename, RNASeqSample)
    μ, σ = approximate_likelihood(sample)

    h5open(output_filename, "w") do out
        n = sample.n
        out["n"] = sample.n
        out["mu", "compress", 1] = μ[1:n]
        out["sigma", "compress", 1] = σ[1:n]
    end
end


function approximate_likelihood_from_isolator(input_filename, output_filename)
    input = open(input_filename)

    n = parse(Int, readline(input))
    #n = 3 # XXX

    m = parse(Int, readline(input))
    @show (m, n)
    I = Array(Int, 0)
    J = Array(Int, 0)
    V = Array(Float32, 0)
    for line in eachline(input)
        j_, i_, v_ = split(line, ',')
        i = 1 + parse(Int, i_)
        j = 1 + parse(Int, j_)
        v = parse(Float32, j_)

        # XXX
        #if j == 72281
            #push!(I, i)
            #push!(J, 1)
            #push!(V, v)

            #push!(I, i)
            #push!(J, 2)
            #push!(V, v)
        #else
            #continue
        #end

        #if j == 72281
            #j = 1
        #elseif j == 72283
            #j = 3
        #else
            #continue
        #end

        push!(I, i)
        push!(J, j)
        push!(V, v)
    end
    X = sparse(I, J, V, m, n)
    #mklX = MKLSparseMatrixCSC(X)
    rsbX = RSBMatrix(X)
    sample = RNASeqSample(m, n, rsbX)

    μ, σ = approximate_likelihood(sample)

    h5open(output_filename, "w") do out
        n = sample.n
        out["n"] = sample.n
        out["mu", "compress", 1] = μ[1:n]
        out["sigma", "compress", 1] = σ[1:n]
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

    gradv = reinterpret(FloatVec, grad)
    hlfv = fill(FloatVec, 0.5f0)
    for i in 1:length(vs)
        gradv[i] += hlfv
    end

    # Note: This is the gradient for σ, not ω
    #gradv = reinterpret(FloatVec, grad)
    #twov = fill(FloatVec, 2.0f0)
    #for i in 1:length(vs)
        #gradv[i] +=  inv(vs[i] .* twov)
    #end

    #vs_sumlogs = sum(mapreduce(log, +, 0.0, σ))
    #entropy = n * log(2 * π * e) + vs_sumlogs
    #for i in 1:length(σ)
        #grad[i] += inv(σ[i])
    #end

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


function approximate_likelihood(s::RNASeqSample)
    m, n = size(s)
    model = Model(m, n)

    # step size constants
    ss_τ = 1.0
    ss_ε = 1e-16
    ss_ω_α = 0.1
    ss_μ_α = 0.01
    ss_η = 1.0
    ss_max_μ_step = 5e-1
    ss_max_ω_step = 1e-1
    srand(4324)

    # number of monte carlo samples to estimate gradients an elbo at each
    # iteration
    num_mc_samples = 2
    η = fillpadded(FloatVec, 0.0f0, n)
    ζ = fillpadded(FloatVec, 0.0f0, n)

    μ = fillpadded(FloatVec, 0.0f0, n)
    σ = fillpadded(FloatVec, 1.0f0, n, 1.f0)
    ω = fillpadded(FloatVec, -3.0f0, n) # log transformed σ

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

    output_idx = 16

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

            lp = log_likelihood(model, s.X, ζ, π_grad)
            @assert isfinite(lp)
            @show lp
            elbo += lp
            μ_grad .+= π_grad
            ω_grad .+= π_grad .* η .* σ
        end
        μ_grad /= num_mc_samples
        ω_grad /= num_mc_samples
        ω_grad .+= 1
        elbo /= num_mc_samples
        elbo += normal_entropy!(ω_grad, σ, n)
        max_elbo = max(max_elbo, elbo)
        @assert isfinite(elbo)
        @printf("ELBO: %e\n", elbo)

        if step_num == 1
            s_μ[:] = μ_grad.^2
            s_ω[:] = ω_grad.^2
        end

        c = ss_η * step_num^(-0.5 + ss_ε)

        for i in 1:length(μ)
            s_μ[i] = (1 - ss_μ_α) * s_μ[i] + ss_μ_α * μ_grad[i]^2
            ρ = c / (ss_τ + sqrt(s_μ[i]))
            μ[i] += clamp(ρ * μ_grad[i], -ss_max_μ_step, ss_max_μ_step)
        end

        for i in 1:length(ω)
            s_ω[i] = (1 - ss_ω_α) * s_ω[i] + ss_ω_α * ω_grad[i]^2
            ρ = c / (ss_τ + sqrt(s_ω[i]))
            ω[i] += clamp(ρ * ω_grad[i], -ss_max_ω_step, ss_max_ω_step)
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

        # TODO: reasonable stopping criteria
        if step_num > 200
            break
        end

        #if small_step_count > max_small_steps ||
           #fruitless_step_count > max_fruitless_steps ||
           #step_num > max_steps
            #break
        #end


        #@show c
        #@show s_μ[output_idx]
        #@show c / (ss_τ + sqrt(s_μ[output_idx]))
        #@show μ[output_idx]
        #@show μ_grad[output_idx]
        #@show ω[output_idx]
        #@show ω_grad[output_idx]
        #@show model.π_simplex[output_idx]


        #@show μ
        #@show μ_grad
        #@show model.π_simplex

        #log_likelihood(model, s.X, ζ, π_grad)
        #@show ζ
        #@show μ
        #@show μ_grad
        #@show model.π_simplex
    end

    #@show step_num

    #idx = searchsorted(ordinalrank(μ), 9 * div(length(μ), 10))
    #@show idx.start

    #samples, weights = diagnostic_samples(model, mklX, μ, σ, idx.start)
    #out = open("diagnostic_samples.csv", "w")
    #for (s, w) in zip(samples, weights)
        #@printf(out, "%.12e,%.12e\n", Float64(s), Float64(w))
    #end
    #close(out)

    map!(exp, σv, ωv)
    return μ, σ
end

