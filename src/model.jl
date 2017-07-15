
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


function log_likelihood_loop1{GRADONLY}(frag_probs::Vector{Float32},
                                        log_frag_probs::Vector{Float32}, m,
                                        ::Type{Val{GRADONLY}})
    Threads.@threads for i in 1:m
        if !GRADONLY
            log_frag_probs[i] = log(frag_probs[i])
        end
        frag_probs[i] = inv(frag_probs[i])
    end
    return sum(log_frag_probs)
end


# assumes a flat prior on π
function log_likelihood{GRADONLY}(model::Model, X, effective_lengths, xs, x_grad,
                                  ::Type{Val{GRADONLY}})
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
    efflen_ladj = 0.0f0
    if !GRADONLY
        for i in 1:n
            efflen_ladj += log(effective_lengths[i])
        end
        efflen_ladj -= n * log(scaled_simplex_sum)
    end

    # conditional fragment probabilities
    A_mul_B!(unsafe_wrap(Vector{Float32}, pointer(frag_probs), m, false), X, ws)
    for k in m+1:length(frag_probs)
        frag_probs[k] = 1.0f0
    end

    # log likelihood
    lp = log_likelihood_loop1(frag_probs, model.log_frag_probs, m, Val{GRADONLY})

    # compute df / dw (where w is effective length weighted mixtures)
    At_mul_B!(x_grad, X, unsafe_wrap(Vector{Float32}, pointer(frag_probs), m, false))

    # @show xs
    # @show ws
    # @show x_grad

    # compute df / dx
    # x_grad[i] now holds df/dw_i where w is the the effective length
    # weighted mixtures. Here we update it to hold df/dx_i, where x are
    # unweighted mixtures.
    c = 0.0
    for i in 1:n
        c += (ws[i] / scaled_simplex_sum) * x_grad[i]
    end

    for i in 1:n-1
        x_grad[i] = (effective_lengths[i] / scaled_simplex_sum) * x_grad[i] -
                    (effective_lengths[n] / scaled_simplex_sum) * x_grad[n] -
                    (effective_lengths[i] - effective_lengths[n]) * c
    end
    x_grad[n] = 0.0

    # compute terms for derivatives of efflen_ladj (log absolute determinant of the jacobian)
    # for the effective length transform
    for i in 1:n-1
        x_grad[i] += (effective_lengths[n] - effective_lengths[i]) / scaled_simplex_sum
    end

    # @show x_grad

    # @show extrema(xs)
    # @show extrema(ws)
    # @show extrema(x_grad)

    @assert isfinite(lp)
    @assert isfinite(efflen_ladj)

    return lp + efflen_ladj
end


