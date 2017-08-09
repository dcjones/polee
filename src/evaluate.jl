




"""
Generate samples from an approximated likelihood stored in the given file.
"""
function sample_likap(input_filename::String, num_samples)
    input = h5open(input_filename)
    mu = read(input["mu"])
    sigma = exp.(read(input["omega"]))
    effective_lengths = read(input["effective_lengths"])
    t = HSBTransform(read(input["node_parent_idxs"]),
                     read(input["node_js"]))
    close(input)
    n = length(mu) + 1

    zs = Array{Float32}(n-1)
    ys = Array{Float64}(n-1)
    xs = Array{Float32}(n)
    samples = Array{Float32}(num_samples, n)
    eps = 1e-10

    for i in 1:num_samples
        for j in 1:n-1
            zs[j] = randn(Float32)
        end
        logit_normal_transform!(mu, sigma, zs, ys, Val{true})
        ys = clamp!(ys, eps, 1 - eps)
        hsb_transform!(t, ys, xs, Val{true})
        xs = clamp!(xs, eps, 1 - eps)

        for j in 1:n
            xs[j] /= effective_lengths[j]
        end
        xs ./= sum(xs)

        samples[i,:] = xs
    end

    return samples
end



"""
Read output from the gibbs sampler.
"""
function read_gibbs_samples(input_filename::String)
    return readcsv(open(input_filename), Float32)
end


"""
Compute feature marginals given a feature dictionary mapping every transcript
numbers to some feature number.
"""
function feature_marginals(samples::Matrix{Float32}, features::Dict{Int, Int})
    # TODO
end


"""
Measure absolute difference in expected log2 fold change between estimates
from the gibbs sampler and estimates from an approximated likelihood.
"""
function log_fold_change_error(a_gibbs, b_gibbs, a_likap, b_likap)
    num_samples, n = size(a_gibbs)
    @assert size(b_gibbs) == (num_samples, n)
    @assert size(a_likap) == (num_samples, n)
    @assert size(b_likap) == (num_samples, n)

    losses = Array{Float32}(n)
    for i in 1:n
        loss = 0.0f0

        # TODO: I should also do this with median/quantiles
        lfc_expect_gibbs = 0.0f0
        lfc_expect_likap = 0.0f0
        for k in 1:num_samples
            lfc_expect_gibbs += log2(a_gibbs[k, i]) - log2(b_gibbs[k, i])
            lfc_expect_likap += log2(a_likap[k, i]) - log2(b_likap[k, i])
            if !isfinite(lfc_expect_gibbs) || !isfinite(lfc_expect_likap)
                @show (a_gibbs[k, i], b_gibbs[k, i], a_likap[k, i], b_likap[k, i])
                exit()
            end
        end
        lfc_expect_gibbs /= num_samples
        lfc_expect_likap /= num_samples

        losses[i] = abs(lfc_expect_gibbs - lfc_expect_likap)
    end

    return losses
end


