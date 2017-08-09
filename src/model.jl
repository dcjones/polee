
type Model
    m::Int # number of fragments
    n::Int # number of transcripts

    # work vector for computing posterior log prob
    frag_probs::Vector{Float32}

    # work for taking log of frag_probs in a multithreaded fashion
    log_frag_probs::Vector{Float32}
end


function Model(m, n)
    return Model(Int(m), Int(n),
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
    m, n = model.m, model.n

    # conditional fragment probabilities
    pAt_mul_B!(frag_probs, Xt, xs)

    # log likelihood
    lp = 0.0
    if !GRADONLY
        log!(model.log_frag_probs, frag_probs, m)
        lp = sum(model.log_frag_probs)
        @assert isfinite(lp)
    end
    inv!(frag_probs, m)

    # compute df / dw (where w is effective length weighted mixtures)
    pAt_mul_B!(x_grad, X, frag_probs)

    return lp
end


