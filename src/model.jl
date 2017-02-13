
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

    # intermediate values in gradient computation
    work1::Vector{Float32}
    work2::Vector{Float32}
    work3::Vector{Float32}
end

function Model(m, n)
    return Model(Int(m), Int(n),
                 fillpadded(FloatVec, 0.0, n),
                 fillpadded(FloatVec, 0.0, m),
                 fillpadded(FloatVec, 0.0, n),
                 fillpadded(FloatVec, 0.0, n),
                 fillpadded(FloatVec, 0.0, n),
                 fillpadded(FloatVec, 0.0, n),
                 fillpadded(FloatVec, 0.0, n),
                 fillpadded(FloatVec, 0.0, n))
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
function simplex!(k, xs, xs_sum, work, zs, ys)
    @assert length(xs) >= k
    @assert length(xs_sum) >= k
    @assert length(work) >= k
    @assert length(zs) >= k

    ladj = 0.0
    xsum = 0.0
    xs_sum[1] = 0.0

    # reusing these vectors as temporary storage
    work1 = xs_sum
    work2 = work

    Threads.@threads for i in 1:k-1
        zs[i] = logistic(Float64(ys[i]) + log(1/(k - i)))
        work1[i+1] = log(zs[i])
        work2[i+1] = log1p(-zs[i])
    end

    for i in 1:k-1
        log_z = work1[i+1]
        log_one_minus_z = work2[i+1]

        ladj += log_z + log_one_minus_z + log(1.0f0 - xsum)
        xs[i] = (1 - xsum) * zs[i]

        xsum += xs[i]
        xs_sum[i+1] = xsum

        if Float32(xsum) >= 1.0 && i < k-1
            @show xsum
            @show xs[i]
            @show ys[i]
            @show zs[i]
        end
        @assert Float32(xsum) < 1.0 || i == k-1
    end

    #for i in 1:k-1
        #zs[i] = logistic(ys[i] + log(1/(k - i)))

        #log_one_minus_z = log(1 - zs[i])
        #ladj += log(zs[i]) + log_one_minus_z + log(1 - xsum)

        #xs[i] = (1 - xsum) * zs[i]

        #xsum += xs[i]
        #xs_sum[i+1] = xsum

        #z_log_sum += log_one_minus_z
        #zs_log_sum[i+1] = z_log_sum
    #end
    xs[k] = 1 - xsum

    #for i in 1:k-1
        #grad[i] +=
            #1 - 2*zs[i] +
            #(1 - i - k) * (1 + xs[i]) * zs[i] * (1 - zs[i])
    #end

    return ladj
end


function simplex_vec!(k, xs, grad, xs_sum, zs, zs_log_sum, ys)
    @assert length(xs) >= k
    @assert length(grad) >= k
    @assert length(xs_sum) >= k
    @assert length(zs) >= k
    @assert length(zs_log_sum) >= k

    xs_sumv = reinterpret(FloatVec, xs_sum)
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

    vecsize = div(sizeof(FloatVec), sizeof(Float32))
    @inbounds for i in 1:length(zvs)
        offset = fill(FloatVec, Float32((i - 1) * vecsize))
        iv = countv + offset
        zvs[i] = FastMath.logistic(yvs[i] + log(inv(kv - iv)))
    end
    zs[k] = 0.0

    # compute xs, xs_sum, zs_log_sum
    xsum = 0.0
    @inbounds for i in 1:k-1
        xs[i] = (1 - xsum) * zs[i]
        xsum += xs[i]
        xs_sum[i+1] = xsum
    end
    xs[k] = 1 - xsum

    @inbounds for i in 1:length(zvs)-1
        log_one_minus_xsum = log(fill(FloatVec, 1.0f0) - xs_sumv[i])
        log_one_minus_z = log(fill(FloatVec, 1.0f0) - zvs[i])
        ladj += log(zvs[i]) + log_one_minus_z + log_one_minus_xsum
        zvs_log_sum[i] = cumsum(log_one_minus_z)
        @show log_one_minus_z
        @show cumsum(log_one_minus_z)
        exit()
    end
    total_ladj = sum(ladj)

    # shift zs_log_sum over by one
    z_log_sum = zs_log_sum[1]
    zs_log_sum[1] = 0.0
    @inbounds for i in 2:length(zs_log_sum)
        zs_log_sum[i], z_log_sum = z_log_sum, zs_log_sum[i]
    end

    @inbounds for i in ((length(zvs)-1)*vecsize+1):k-1
        total_ladj += log(zs[i]) + log(1 - zs[i]) + log(1 - xs_sum[i])
        zs_log_sum[i] = zs_log_sum[i-1] + log(1 - zs[i-1])
    end
    if k > 1
        zs_log_sum[k] = zs_log_sum[k-1] + log(1 - zs[k-1])
    end

    # compute grad
    gradv = reinterpret(FloatVec, grad)
    @inbounds for i in 1:length(gradv)
        offset = fill(FloatVec, Float32((i - 1) * vecsize))
        iv = countv + offset
        gradv[i] =
            onev - fill(FloatVec, 2.0f0) .* zvs[i] +
            (onev - iv - kv) .* (onev + xvs[i]) .* zvs[i] .* (onev - zvs[i])
    end
    grad[k] = 0.0

    @show zs[1:10]
    @show zs_log_sum[1:10]
    exit()

    # return horizontal sum of ladj
    return total_ladj
end


# assumes a flat prior on π
function log_likelihood(model::Model, X, π, grad)
    frag_probs = model.frag_probs
    fill!(grad, 0.0)
    m, n = model.m, model.n

    # transform π to simplex
    ladj = simplex!(model.n, model.π_simplex, model.xs_sum,
                    model.work1, model.zs, π)
    #@show model.π_simplex[1:n]
    #ladj = simplex_vec!(model.n, model.π_simplex, grad, model.xs_sum, model.zs,
                        #model.zs_log_sum, π)
    #@show model.π_simplex[16]

    # conditional fragment probabilities
    #A_mul_B!(view(frag_probs, 1:m), X, view(model.π_simplex, 1:n))
    A_mul_B!(frag_probs, X, model.π_simplex)

    # log-likelihood
    lpv = fill(FloatVec, 0.0f0)
    frag_probs_v = reinterpret(FloatVec, frag_probs)
    for i in 1:length(frag_probs_v)
        lpv += log(frag_probs_v[i])
        frag_probs_v[i] = inv(frag_probs_v[i])
    end
    lp = sum(lpv)

    # computed untransformed gradient in raw_grad
    raw_grad = model.raw_grad
    #At_mul_B!(view(raw_grad, 1:n), X, view(frag_probs, 1:m))
    At_mul_B!(raw_grad, X, frag_probs)

    # compute gradient of simplex transform
    zs = model.zs
    xs_sum = model.xs_sum

    log1mz_sum = model.work1
    log1mz_sum[1] = log(1.0f0 - zs[1])
    for i in 2:n
        log1mz_sum[i] = log1mz_sum[i-1] + log(1.0f0 - zs[i])
    end

    us = model.work2
    us[1] = zs[1] * raw_grad[1]
    for i in 2:n
        us[i] = us[i-1] + zs[i] * exp(log1mz_sum[i-1]) * raw_grad[i]
    end

    # now we can compute gradients

    #@show raw_grad[1:n]
    #@show log1mz_sum[1:n]
    #@show us[1:n]
    #@show zs[1:n]

    for i in 1:model.n-1
        dxi_dyi = (1.0 - xs_sum[i]) * zs[i] * (1.0 - zs[i])

        # (dp(x) / dyi)_i
        df_dyi_i = raw_grad[i] * dxi_dyi
        grad[i] += df_dyi_i

        # (dp(x) / dyi)_n
        dxn_dxi = -exp(log1mz_sum[n-1] - log1mz_sum[i])
        df_dyi_n = dxi_dyi * dxn_dxi * raw_grad[n]
        grad[i] += df_dyi_n

        # I feel pretty confident about these first two parts
        # I guess the third part is mots iffy

        # sum_{j=i+1}^{n-1} (dp(x) / dyi)_j
        df_dy_mid = -dxi_dyi * (us[n-1] - us[i]) * exp(-log1mz_sum[i])
        grad[i] += df_dy_mid

        #@show (i, df_dyi_i, df_dyi_n, df_dy_mid)
        @assert isfinite(grad[i])
    end


    # gradient of the log absolute determinate of the jacobian (ladj)
    us[1] = -1 / (1 - xs_sum[1])
    for i in 2:n-1
        us[i] = us[i-1] - exp(log1mz_sum[i-1]) / (1 - xs_sum[i])
        if !isfinite(us[i])
            @show us[i-1]
            @show exp(log1mz_sum[i-1])
            @show (1 - xs_sum[i])
            @show xs_sum[i]
            @show i
        end
        @assert isfinite(us[i])
    end

    for i in 1:n-1
        # d/dy_i log(dx_i / dy_i)
        grad[i] += (1 - 2*zs[i])

        dxi_dyi = (1.0 - xs_sum[i]) * zs[i] * (1.0 - zs[i])
        grad[i] += dxi_dyi * (us[n-1] - us[i]) * exp(-log1mz_sum[i])
        if !isfinite(grad[i])
            @show dxi_dyi
            @show us[n-1]
            @show us[i]
            @show log1mz_sum[i]
            @show exp(-log1mz_sum[i])
        end
        @assert isfinite(grad[i])
    end

    #@show grad[1:n]

    return lp + ladj
end


