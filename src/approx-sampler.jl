


mutable struct ApproxLikelihoodSampler
    zs::Vector{Float32}
    ys::Vector{Float64}
    mu::Vector{Float32}
    sigma::Vector{Float32}
    alpha::Vector{Float32}
    t::PolyaTreeTransform

    function ApproxLikelihoodSampler()
        return new(
            Float32[], Float64[], Float32[], Float32[], Float32[])
    end
end


function set_transform!(
        als::ApproxLikelihoodSampler,
        t::PolyaTreeTransform,
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
    sinh_asinh_transform!(als.alpha, als.zs, als.zs)
    logit_normal_transform!(als.mu, als.sigma, als.zs, als.ys)
    transform!(als.t, als.ys, xs)
end


"""
Compute element-wise quantiles of the approximated likelihood.
"""
function Statistics.quantile(
        loaded_samples::LoadedSamples,
        transforms::Vector{PolyaTreeTransform}, qs=(0.01, 0.99), N=100)

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


function posterior_mean(loaded_samples::LoadedSamples, N=100)
    num_samples, n = size(loaded_samples.x0_values)
    als = ApproxLikelihoodSampler()

    n = size(loaded_samples.x0_values, 2)
    pm = similar(loaded_samples.x0_values)
    fill!(pm, 0.0f0)
    xs = Array{Float32}(undef, n)

    for i in 1:num_samples
        input = h5open(loaded_samples.sample_filenames[i])
        node_parent_idxs = read(input["node_parent_idxs"])
        node_js          = read(input["node_js"])
        t = PolyaTreeTransform(node_parent_idxs, node_js)
        close(input)

        set_transform!(
            als, t,
            loaded_samples.la_mu_values[i,:],
            loaded_samples.la_sigma_values[i,:],
            loaded_samples.la_alpha_values[i,:])

        for k in 1:N
            rand!(als, xs)
            clamp!(xs, 1f-15, 0.9999999f0)
            pm[i,:] .+= xs;
        end
    end
    pm ./= N;

    return pm
end

