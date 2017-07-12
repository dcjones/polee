
type Model
    m::Int # number of fragments
    n::Int # number of transcripts

    # xs transformed to the effective length weighted simplex
    ws::Vector{Float32}

    # work vector for computing posterior log prob
    frag_probs::Vector{Float32}

    # work for taking log of frag_probs in a multithreaded fashion
    log_frag_probs::Vector{Float32}
end


function Model(m, n)
    return Model(Int(m), Int(n),
                 Vector{Float32}(n),
                 fillpadded(FloatVec, 0.0f0, m, 1.0f0),
                 fillpadded(FloatVec, 0.0f0, m, 1.0f0))
end


# function log_likelihood_loop1(frag_probs_v)
#     lpv = fill(fill(FloatVec, 0.0f0), Threads.nthreads())
#     Threads.@threads for i in 1:length(frag_probs_v)
#         lpv[Threads.threadid()] += log(frag_probs_v[i])
#         frag_probs_v[i] = inv(frag_probs_v[i])
#     end
#     ans = 0.0
#     for v in lpv
#         ans += sum(v)
#     end
#     return ans
# end


function log_likelihood_loop1(frag_probs::Vector{Float32}, log_frag_probs::Vector{Float32}, m)
    Threads.@threads for i in 1:m
        log_frag_probs[i] = log(frag_probs[i])
        frag_probs[i] = inv(frag_probs[i])
    end
    return sum(log_frag_probs)
end


# assumes a flat prior on π
function log_likelihood(model::Model, X, effective_lengths, xs, x_grad)
    frag_probs = model.frag_probs
    ws = model.ws
    m, n = model.m, model.n

    # transform to effective length adjusted simplex
    scaled_simplex_sum = 0.0
    for i in 1:n
        scaled_simplex_sum += effective_lengths[i] * xs[i]
    end

    for i in 1:n
        ws[i] = effective_lengths[i] * xs[i] / scaled_simplex_sum
    end

    # log jacobian determinate for the effective length transform
    efflen_ladj = 0.0
    for i in 1:n
        efflen_ladj += log(effective_lengths[i])
    end
    efflen_ladj -= n * log(scaled_simplex_sum)

    # conditional fragment probabilities
    A_mul_B!(unsafe_wrap(Vector{Float32}, pointer(frag_probs), m, false), X, ws)
    for k in m+1:length(frag_probs)
        frag_probs[k] = 1.0f0
    end

    # log-likelihood
    # frag_probs_v = reinterpret(FloatVec, frag_probs)
    # lp = log_likelihood_loop1(frag_probs_v)

    # `log_likelihood_loop1` does the below but realy realy fast
    lp = log_likelihood_loop1(frag_probs, model.log_frag_probs, m)
    # lp = 0.0
    # for i in 1:m
    #     lp += log(frag_probs[i])
    #     frag_probs[i] = inv(frag_probs[i])
    # end

    # compute df / dw (where w is effective length weighted mixtures)
    At_mul_B!(x_grad, X, unsafe_wrap(Vector{Float32}, pointer(frag_probs), m, false))

    # compute df / dx
    # x_grad[i] now holds df/dw_i where w is the the effective length
    # weighted mixtures. Here we update it to hold df/dx_i, where x are
    # unweighted mixtures.
    c = 0.0
    for i in 1:n
        c += (ws[i] / scaled_simplex_sum) * x_grad[i]
    end

    for i in 1:n
        x_grad[i] = effective_lengths[i] * (x_grad[i] / scaled_simplex_sum - c);
    end

    # compute terms for derivatives of efflen_ladj (log absolute determinant of the jacobian)
    # for the effective length transform
    for i in 1:n-1
        x_grad[i] += n * (effective_lengths[n] - effective_lengths[i]) / scaled_simplex_sum
    end

    @assert isfinite(lp)
    @assert isfinite(efflen_ladj)

    return lp + efflen_ladj
end


