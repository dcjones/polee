
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
    xs_sum::Vector{Float64}
    zs::Vector{Float64}

    # intermediate values in gradient computation
    log1mz_sum::Vector{Float64}
    onemz_prod::Vector{Float64}
    work::Vector{Float64}
end

function Model(m, n)
    return Model(Int(m), Int(n),
                 fillpadded(FloatVec, 0.0, n),
                 fillpadded(FloatVec, 0.0, m),
                 fillpadded(FloatVec, 0.0, n),
                 fill(0.0, n),
                 fill(0.0, n),
                 fill(0.0, n),
                 fill(0.0, n),
                 fill(0.0, n))
end


"""
Logistic (inverse logit) function.
"""
logistic(x) = 1 / (1 + exp(-x))


function simplex!_loop1(k, xs, work1, work2, zs, ys)
    Threads.@threads for i in 1:k-1
        zs[i] = logistic(ys[i] + log(1.0f0/(k - i)))
        work1[i+1] = log(zs[i])
        work2[i+1] = log1p(-zs[i])
    end
end


function simplex!_loop2(k, log1mz_sum, onemz_prod)
    Threads.@threads for i in 1:k-1
        onemz_prod[i] = exp(log1mz_sum[i])
    end
end


"""
Transform an unconstrained vector `ys` to a simplex.

Return the log absolute determinate of the jacobian (ladj), and store the
gradient of the ladj in `grad`.

Some intermediate values are also stored:
    * `xs_sum`: cumulative sum of xs where `xs_sum[i]` is the sum of all
                `xs[j]` for `j < i`.
    * `zs`: intermediate adjusted y-values
    * `log1mz_sum`: cumulative sum of `log(1 - zs[j])`.
    * `onemz_prod`: cumulative product of `(1 - zs[j])`.
"""
function simplex!(k, xs, xs_sum, work, zs, log1mz_sum, onemz_prod, ys)
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

    simplex!_loop1(k, xs, work1, work2, zs, ys)

    log1mz_sum[1] = work2[2]
    onemz_prod[1] = 1.0 - zs[1]
    for i in 2:k-1
        log1mz_sum[i] = log1mz_sum[i-1] + work2[i+1]
    end
    simplex!_loop2(k, log1mz_sum, onemz_prod)

    for i in 1:k-1
        log_z = work1[i+1]
        log_one_minus_z = work2[i+1]
        log_one_minus_xsum = log1p(-xsum)
        ladj += log_z + log_one_minus_z + log_one_minus_xsum

        xs[i] = (1 - xsum) * zs[i]
        xsum += xs[i]
        xs_sum[i+1] = xsum
        @assert xs_sum[i+1] <= 1.0
    end
    xs[k] = 1 - xsum

    return ladj
end


function log_likelihood_loop1(frag_probs_v)
    lpv = fill(fill(FloatVec, 0.0f0), Threads.nthreads())
    Threads.@threads for i in 1:length(frag_probs_v)
        lpv[Threads.threadid()] += log(frag_probs_v[i])
        frag_probs_v[i] = inv(frag_probs_v[i])
    end
    ans = 0.0
    for v in lpv
        ans += sum(v)
    end
    return ans
end


#function log_likelihood_loop2(n, xs_sum, zs, raw_grad, onemz_prod, us, grad)
    #Threads.@threads for i in 1:n-1
        #dxi_dyi = (1 - xs_sum[i]) * zs[i] * (1 - zs[i])

        ## (dp(x) / dyi)_i
        #df_dyi_i = raw_grad[i] * dxi_dyi
        #grad[i] += df_dyi_i

        ## (dp(x) / dyi)_n
        ##dxn_dxi = -exp(log1mz_sum[n-1] - log1mz_sum[i])
        #dxn_dxi = -onemz_prod[n-1] / onemz_prod[i]
        #df_dyi_n = dxi_dyi * dxn_dxi * raw_grad[n]
        #grad[i] += df_dyi_n

        ## sum_{j=i+1}^{n-1} (dp(x) / dyi)_j
        #df_dy_mid = -dxi_dyi * (us[n-1] - us[i]) / onemz_prod[i]
        #grad[i] += df_dy_mid
    #end
#end


# assumes a flat prior on π
function log_likelihood(model::Model, X, effective_lengths, π, grad)
    frag_probs = model.frag_probs
    fill!(grad, 0.0)
    m, n = model.m, model.n

    log1mz_sum = model.log1mz_sum
    onemz_prod = model.onemz_prod

    # transform π to simplex
    ladj = simplex!(model.n, model.π_simplex, model.xs_sum,
                    model.work, model.zs, log1mz_sum, onemz_prod, π)

    # transform to effect length adjusted simplex
    scaled_simplex_sum = 0.0
    for i in 1:n
        scaled_simplex_sum += effective_lengths[i] * model.π_simplex[i]
    end

    for i in 1:n
        model.π_simplex[i] =
            effective_lengths[i] * model.π_simplex[i] / scaled_simplex_sum
    end

    # conditional fragment probabilities
    A_mul_B!(frag_probs, X, model.π_simplex)

    # log-likelihood
    frag_probs_v = reinterpret(FloatVec, frag_probs)

    lp = log_likelihood_loop1(frag_probs_v)
    #for i in 1:length(frag_probs_v)
        #lpv += log(frag_probs_v[i])
        #frag_probs_v[i] = inv(frag_probs_v[i])
    #end
    #lp = sum(lpv)

    # computed untransformed gradient in raw_grad
    raw_grad = model.raw_grad
    At_mul_B!(raw_grad, X, frag_probs)

    # Gradients with respect to simplex(x)
    c = 0.0
    for i in 1:model.n
        c += (model.π_simplex[i] / scaled_simplex_sum) * raw_grad[i]
    end

    for i in 1:model.n
        raw_grad[i] =
            effective_lengths[i] * (raw_grad[i] / scaled_simplex_sum - c);
    end

    # compute gradient of simplex transform
    zs = model.zs
    xs_sum = model.xs_sum

    us = model.work
    us[1] = zs[1] * raw_grad[1]
    for i in 2:n
        us[i] = us[i-1] + zs[i] * onemz_prod[i-1] * raw_grad[i]
    end

    # compute df(x) / dyi gradients
    for i in 1:model.n-1
        dxi_dyi = (1 - xs_sum[i]) * zs[i] * (1 - zs[i])

        # (dp(x) / dyi)_i
        df_dyi_i = raw_grad[i] * dxi_dyi
        grad[i] += df_dyi_i

        # (dp(x) / dyi)_n
        #dxn_dxi = -exp(log1mz_sum[n-1] - log1mz_sum[i])
        dxn_dxi = -onemz_prod[n-1] / onemz_prod[i]
        df_dyi_n = dxi_dyi * dxn_dxi * raw_grad[n]
        grad[i] += df_dyi_n

        # sum_{j=i+1}^{n-1} (dp(x) / dyi)_j
        df_dy_mid = -dxi_dyi * (us[n-1] - us[i]) / onemz_prod[i]
        grad[i] += df_dy_mid
        #if !isfinite(grad[i])
            #@show (dxi_dyi, us[n-1], us[i], onemz_prod[i])
            #exit()
        #end
    end
    #log_likelihood_loop2(n, xs_sum, zs, raw_grad, onemz_prod, us, grad)

    # gradient of the log absolute determinate of the jacobian (ladj)
    us[1] = -1 / (1 - xs_sum[1])
    for i in 2:n-1
        us[i] = us[i-1] - onemz_prod[i-1] / (1 - xs_sum[i])
    end

    for i in 1:n-1
        # d/dy_i log(dx_i / dy_i)
        grad[i] += (1 - 2*zs[i])

        # d/dy_i sum_{k=i+1}^{n-1} log(dx_k / dy_i)
        dxi_dyi = (1.0 - xs_sum[i]) * zs[i] * (1.0 - zs[i])
        grad[i] += dxi_dyi * (us[n-1] - us[i]) / onemz_prod[i]
        #if !isfinite(grad[i])
            #@show (dxi_dyi, us[n-1], us[i], onemz_prod[i])
            #exit()
        #end
    end

    @assert isfinite(lp)
    @assert isfinite(ladj)

    return lp + ladj
end


