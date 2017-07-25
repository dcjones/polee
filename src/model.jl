
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
                 Vector{Float32}(m),
                 Vector{Float32}(m))
end


function log!{T}(ys::Vector{T}, xs::Vector{T}, m)
    Threads.@threads for i in 1:m
        ys[i] = log(xs[i])
    end
end


function inv!{T}(xs::Vector{T}, m)
    Threads.@threads for i in 1:m
        xs[i] = one(T) / xs[i]
    end
end


# assumes a flat prior on π
function log_likelihood{GRADONLY}(model::Model, X, Xt, effective_lengths, xs, x_grad,
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
    efflen_ladj = 0.0
    if !GRADONLY
        for i in 1:n
            efflen_ladj += log(effective_lengths[i])
        end
        efflen_ladj -= n * log(scaled_simplex_sum)
    end

    # conditional fragment probabilities
    pAt_mul_B!(unsafe_wrap(Vector{Float32}, pointer(frag_probs), m, false), Xt, ws)

    # log likelihood
    lp = 0.0
    if !GRADONLY
        log!(model.log_frag_probs, frag_probs, m)
        lp = sum(model.log_frag_probs)
    end
    inv!(frag_probs, m)

    # compute df / dw (where w is effective length weighted mixtures)
    pAt_mul_B!(x_grad, X, unsafe_wrap(Vector{Float32}, pointer(frag_probs), m, false))

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
        x_grad[i] += n * (effective_lengths[n] - effective_lengths[i]) / scaled_simplex_sum
    end

    @assert isfinite(lp)
    @assert isfinite(efflen_ladj)

    return lp + efflen_ladj
end


