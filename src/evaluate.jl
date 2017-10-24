

"""
"""
function sample_likap(input_filename::String, num_samples)
    input = h5open(input_filename)
    approx_type_name = read(attrs(input["metadata"])["approximation"])
    approx_type = eval(parse(approx_type_name))
    return sample_likap(approx_type, input, num_samples)
end


"""
Generate samples from likelihood approximated with a multivariate logistic-normal
"""
function sample_likap(approx_type::Type{LogisticNormalApprox},
                      input, num_samples)
    mu = read(input["mu"])
    sigma = exp.(read(input["omega"]))
    effective_lengths = read(input["effective_lengths"])
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
            ys[j] = zs[j] * sigma[j] + mu[j]
        end

        # transform to multivariate logit-normal
        xs[n] = 1.0f0
        exp_y_sum = 1.0f0
        for j in 1:n-1
            xs[j] = exp(ys[j])
            exp_y_sum += xs[j]
        end
        for j in 1:n
            xs[j] /= exp_y_sum
        end
        xs = clamp!(xs, eps, 1 - eps)

        for j in 1:n
            xs[j] /= effective_lengths[j]
        end
        xs ./= sum(xs)

        samples[i,:] = xs
    end

    return samples
end


function sample_likap(approx_type::Type{LogitNormalHSBApprox},
                      input, num_samples)
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


function sample_likap(approx_type::Type{LogitSkewNormalHSBApprox},
                      input, num_samples)
    mu = read(input["mu"])
    sigma = exp.(read(input["omega"]))
    alpha = read(input["alpha"])

    effective_lengths = read(input["effective_lengths"])
    t = HSBTransform(read(input["node_parent_idxs"]),
                     read(input["node_js"]))
    close(input)
    n = length(mu) + 1

    zs0 = Array{Float32}(n-1)
    zs = Array{Float32}(n-1)
    ys = Array{Float64}(n-1)
    xs = Array{Float32}(n)
    samples = Array{Float32}(num_samples, n)
    eps = 1e-10

    for i in 1:num_samples
        for j in 1:n-1
            zs0[j] = randn(Float32)
        end

        sinh_asinh_transform!(alpha, zs0, zs, Val{true})
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
Generate samples from an approximated likelihood stored in the given file.
"""
function sample_likap(approx_type::Type{KumaraswamyHSBApprox},
                      input, num_samples)
    as = exp.(read(input["alpha"]))
    bs = exp.(read(input["beta"]))
    effective_lengths = read(input["effective_lengths"])
    t = HSBTransform(read(input["node_parent_idxs"]),
                     read(input["node_js"]))
    close(input)
    n = length(as) + 1

    zs = Array{Float32}(n-1)
    ys = Array{Float64}(n-1)
    xs = Array{Float32}(n)
    work = zeros(Float32, n-1)
    samples = Array{Float32}(num_samples, n)

    minz = eps(Float32)
    maxz = 1.0f0 - eps(Float32)

    for i in 1:num_samples
        for j in 1:n-1
            zs[j] = min(maxz, max(minz, rand()))
        end
        kumaraswamy_transform!(as, bs, zs, ys, work, Val{true})  # z -> y
        ys = clamp!(ys, LIKAP_Y_EPS, 1 - LIKAP_Y_EPS)

        hsb_transform!(t, ys, xs, Val{true})
        xs = clamp!(xs, LIKAP_Y_EPS, 1 - LIKAP_Y_EPS)

        for j in 1:n
            xs[j] /= effective_lengths[j]
        end
        xs ./= sum(xs)

        samples[i,:] = xs
    end

    return samples
end


function sample_likap(approx_type::Type{NormalILRApprox},
                      input, num_samples)

    mu = read(input["mu"])
    sigma = exp.(read(input["omega"]))
    effective_lengths = read(input["effective_lengths"])
    t = ILRTransform(read(input["node_parent_idxs"]),
                     read(input["node_js"]))
    close(input)
    n = length(mu) + 1
    eps = 1e-10

    zs = Array{Float32}(n-1)
    ys = Array{Float64}(n-1)
    xs = Array{Float32}(n)
    samples = Array{Float32}(num_samples, n)

    for i in 1:num_samples
        for j in 1:n-1
            zs[j] = randn(Float32)
            ys[j] = mu[j] + sigma[j] * zs[j]
        end

        ilr_ladj = ilr_transform!(t, ys, xs, Val{true})                     # y -> x
        xs = clamp!(xs, eps, 1 - eps)

        for j in 1:n
            xs[j] /= effective_lengths[j]
        end
        xs ./= sum(xs)

        samples[i,:] = xs
    end

    return samples
end


function sample_likap(approx_type::Type{NormalALRApprox},
                      input, num_samples)

    mu = read(input["mu"])
    sigma = exp.(read(input["omega"]))
    effective_lengths = read(input["effective_lengths"])
    t = ALRTransform(read(input["refidx"])[1])
    close(input)
    n = length(mu) + 1
    eps = 1e-10

    zs = Array{Float32}(n-1)
    ys = Array{Float64}(n-1)
    xs = Array{Float32}(n)
    samples = Array{Float32}(num_samples, n)

    for i in 1:num_samples
        for j in 1:n-1
            zs[j] = randn(Float32)
            ys[j] = mu[j] + sigma[j] * zs[j]
        end

        ilr_ladj = alr_transform!(t, ys, xs, Val{true})                     # y -> x
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
"""
function read_transcript_gene_nums(genes_db_filename)
    db = SQLite.DB(genes_db_filename)

    results = SQLite.query(db, "select transcript_num, gene_num from transcripts")
    transcript_gene_num = Dict{Int, Int}()
    for (transcript_num, gene_num) in zip(results[:transcript_num], results[:gene_num])
        transcript_gene_num[get(transcript_num)] = get(gene_num)
    end

    return transcript_gene_num
end


"""
Compute feature marginals given a feature dictionary mapping every transcript
numbers to some feature number.
"""
function feature_marginals(samples::Matrix{Float32}, features::Dict{Int, Int})
    num_samples, n1 = size(samples)
    n2 = maximum(values(features))

    marginal_samples = zeros(Float32, num_samples, n2)
    for i in 1:num_samples, j in 1:n1
        marginal_samples[i, features[j]] += samples[i, j]
    end

    return marginal_samples
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

        losses[i] = lfc_expect_likap - lfc_expect_gibbs
    end

    return losses
end


