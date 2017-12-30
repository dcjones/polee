
type Model
    m::Int # number of fragments
    n::Int # number of transcripts

    # work vector for computing posterior log prob
    frag_probs::Vector{Float64}

    # work for taking log of frag_probs in a multithreaded fashion
    log_frag_probs::Vector{Float64}
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
function log_likelihood{GRADONLY}(frag_probs, log_frag_probs, X, Xt, xs, x_grad,
                                  ::Type{Val{GRADONLY}})
    m, n = size(X)

    # conditional fragment probabilities
    pAt_mul_B!(frag_probs, Xt, xs)

    # log likelihood
    lp = 0.0
    if !GRADONLY
        log!(log_frag_probs, frag_probs, m)
        lp = sum(log_frag_probs)
        @assert isfinite(lp)
    end

    pAt_mulinv_B!(x_grad, X, frag_probs)

    return lp
end


