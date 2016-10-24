
using FastMath

type Model
    m::Int # number of fragments
    n::Int # number of transcripts

    # π transformed to the simplex
    π_simplex::Vector{Float32}

    # work vector for computing posterior log prob
    frag_probs::Vector{Float32}

    # intermediate gradient (before accounting for transform)
    raw_grad::Vector{Float32}

    # intermediate values used in simplex calculation
    xs_sum::Vector{Float32}
    zs::Vector{Float32}
    zs_log_sum::Vector{Float32}

    # intermediate values in gradient computation
    grad_work::Vector{Float32}
end

function Model(m, n)
    # round up to align with 8-element vectors for simd
    m_ = 8 * (div(m - 1, 8) + 1)
    n_ = 8 * (div(n - 1, 8) + 1)

    return Model(Int(m), Int(n), zeros(Float32, n_), zeros(Float32, m_),
                 zeros(Float32, n_), zeros(Float32, n_), zeros(Float32, n_),
                 zeros(Float32, n_), zeros(Float32, n_))
end


"""
Logistic (inverse logit) function.
"""
logistic(x) = 1 / (1 + exp(-x))


"""
Transform an unconstrained vector `ys` to a simplex.

Return the log absolute determinate of the jacobian (ladj), and store the
gradient of the ladj in `grad`.

Some intermediate values are also stored:
    * `xs_sum`: cumulative sum of xs where `xs_sum[i]` is the sum of all
                `xs[j]` for `j < i`.
    * `zs`: intermediate adjusted y-values
    * `zs_log_sum`: cumulative sum of `log(1 - zs[j])`.
"""
function simplex!(k, xs, grad, xs_sum, zs, zs_log_sum, ys)
    @assert length(xs) >= k
    @assert length(grad) >= k
    @assert length(xs_sum) >= k
    @assert length(zs) >= k
    @assert length(zs_log_sum) >= k

    ladj = 0.0
    xsum = 0.0
    z_log_sum = 0.0
    xs_sum[1] = 0.0
    zs_log_sum[1] = 0.0

    for i in 1:k-1
        zs[i] = logistic(ys[i] + log(1/(k - i)))

        log_one_minus_z = log(1 - zs[i])
        ladj += log(zs[i]) + log_one_minus_z + log(1 - xsum)

        xs[i] = (1 - xsum) * zs[i]

        xsum += xs[i]
        xs_sum[i+1] = xsum

        z_log_sum += log_one_minus_z
        zs_log_sum[i+1] = z_log_sum
    end
    xs[k] = 1 - xsum

    for i in 1:k-1
        grad[i] +=
            1 - 2*zs[i] +
            (1 - i - k) * (1 + xs[i]) * zs[i] * (1 - zs[i])
    end

    return ladj
end


function simplex_vec!(k, xs, grad, xs_sum, zs, zs_log_sum, ys)
    @assert length(xs) >= k
    @assert length(grad) >= k
    @assert length(xs_sum) >= k
    @assert length(zs) >= k
    @assert length(zs_log_sum) >= k

    xvs = reinterpret(FloatVec, xs)
    yvs = reinterpret(FloatVec, ys)
    zvs = reinterpret(FloatVec, zs)
    zvs_log_sum = reinterpret(FloatVec, zs_log_sum)
    gradv = reinterpret(FloatVec, grad)
    kv = fill(FloatVec, Float32(k))

    ladj = fill(FloatVec, 0.0f0)
    z_log_sum = fill(FloatVec, 0.0f0)
    onev = fill(FloatVec, 1.0f0)
    countv = count(FloatVec)

    # compute zs and zs_log_sum
    @inbounds for i in 1:length(zvs)
        offset = fill(FloatVec, Float32((i - 1) * div(sizeof(FloatVec), sizeof(Float32))))
        iv = countv + offset

        zvs[i] = FastMath.logistic(yvs[i] + log(inv(kv - iv)))

        log_one_minus_z = log(fill(FloatVec, 1.0f0) - zvs[i])
        ladj += log(zvs[i]) + log_one_minus_z

        zvs_log_sum[i] = cumsum(log_one_minus_z)
    end

    # shift zs_log_sum over by one
    z_log_sum = zs_log_sum[1]
    zs_log_sum[1] = 0.0
    @inbounds for i in 2:length(zs_log_sum)
        zs_log_sum[i], z_log_sum = z_log_sum, zs_log_sum[i]
    end

    # compute xs, xs_sum, zs_log_sum
    xsum = 0.0
    @inbounds for i in 1:k-1
        xs[i] = (1 - xsum) * zs[i]
        xsum += xs[i]
        xs_sum[i+1] = xsum
    end

    # compute grad
    gradv = reinterpret(FloatVec, grad)
    @inbounds for i in 1:length(gradv)
        offset = fill(FloatVec, Float32((i - 1) * div(sizeof(FloatVec), sizeof(Float32))))
        iv = countv + offset
        gradv[i] =
            onev - fill(FloatVec, 2.0f0) .* zvs[i] +
            (onev - iv - kv) .* (onev - xvs[i]) .* zvs[i] .* (onev - zvs[i])
    end

    # return horizontal sum of ladj
    return sum(ladj)
end


# assumes a flat prior on π
function log_post(model::Model, X, π, grad)
    frag_probs = model.frag_probs
    fill!(grad, 0.0)

    # transform π to simplex
    ladj = simplex!(model.n, model.π_simplex, grad, model.xs_sum, model.zs,
                    model.zs_log_sum, π)
    #ladj = simplex_vec!(model.n, model.π_simplex, grad, model.xs_sum, model.zs,
                        #model.zs_log_sum, π)

    # conditional fragment probabilities
    A_mul_B!(frag_probs, X, model.π_simplex)

    # log-likelihood
    lpv = fill(FloatVec, 0.0f0)
    frag_probs_v = reinterpret(FloatVec, frag_probs)
    for i in 1:length(frag_probs_v)
        lpv += log(frag_probs_v[i])
        frag_probs_v[i] = inv(frag_probs_v[i])
    end
    lp = sum(lpv)

    # gradients
    raw_grad = model.raw_grad
    At_mul_B!(raw_grad, X, frag_probs)

    # compute the gradients the correct but intractable way
    zs = model.zs
    zs_log_sum = model.zs_log_sum
    xs_sum = model.xs_sum
    grad_work = model.grad_work

    grad_work0 = 0.0
    for j in 1:model.n
        grad_work0 = grad_work[j] =
            grad_work0 + raw_grad[j] * -zs[j] * exp(zs_log_sum[j])
    end

    for i in 1:model.n-1
        b = (grad_work[model.n] - grad_work[i]) * exp(-zs_log_sum[i+1])
        grad[i] += (raw_grad[i] + b) * zs[i] * (1 - zs[i]) * (1 - xs_sum[i])
    end

    @show lp + ladj
    return lp + ladj
end


