


mutable struct ApproxLikelihoodSampler
    zs::Vector{Float32}
    ys::Vector{Float64}
    mu::Vector{Float32}
    sigma::Vector{Float32}
    alpha::Vector{Float32}
    t::HSBTransform

    function ApproxLikelihoodSampler()
        return new(
            Float32[], Float64[], Float32[], Float32[], Float32[])
    end
end


function set_transform!(
        als::ApproxLikelihoodSampler,
        t::HSBTransform,
        mu::Vector{Float32},
        sigma::Vector{Float32},
        alpha::Vector{Float32})
    als.t = t
    als.mu = mu
    als.sigma = sigma
    als.alpha = alpha
    n = length(als.mu) + 1
    if length(als.zs) != n - 1
        als.zs = Vector{Float32}(undef, n - 1)
        als.ys = Vector{Float64}(undef, n - 1)
    end
end


function Random.rand!(als::ApproxLikelihoodSampler, xs::AbstractArray)
    for j in 1:length(als.zs)
        als.zs[j] = randn(Float32)
    end
    sinh_asinh_transform!(als.alpha, als.zs, als.zs, Val{true})
    logit_normal_transform!(als.mu, als.sigma, als.zs, als.ys, Val{true})
    hsb_transform!(als.t, als.ys, xs, Val{true})
end


"""
Compute element-wise quantiles of the approximated likelihood.
"""
function Statistics.quantile(
        loaded_samples::LoadedSamples,
        transforms::Vector{HSBTransform}, qs=(0.01, 0.99), N=100)

    num_samples, n = size(loaded_samples.x0_values)
    als = ApproxLikelihoodSampler()

    all_samples = Array{Float32}(undef, (N, n))
    quantiles = Array{Float32}(undef, (length(qs), num_samples, n))
    tmp = [Array{Float32}(undef, N) for _ in 1:Threads.nthreads()]

    for i in 1:num_samples
        set_transform!(
            als, transforms[i],
            loaded_samples.la_mu_values[i,:],
            loaded_samples.la_sigma_values[i,:],
            loaded_samples.la_alpha_values[i,:])

        for k in 1:N
            rand!(als, view(all_samples, k, :))
        end

        Threads.@threads for j in 1:n
            thrd = Threads.threadid()
            tmp[thrd][:] = all_samples[:, j]
            sort!(tmp[thrd])
            for (k, q) in enumerate(qs)
                quantiles[k, i, j] = quantile(tmp[thrd], q, sorted=true)
            end
        end
    end

    return quantiles
end


