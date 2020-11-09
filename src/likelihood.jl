
struct Model
    m::Int # number of fragments
    n::Int # number of transcripts

    # TODO: Can these be Float32?

    # work vector for computing posterior log prob
    frag_probs::Vector{Float64}

    # work for taking log of frag_probs in a multithreaded fashion
    log_frag_probs::Vector{Float64}
end


function Model(m, n)
    return Model(Int(m), Int(n),
                 zeros(Float32, m),
                 zeros(Float32, m))
end


function log!(ys::Vector{T}, xs::Vector{T}, m) where {T}
    Threads.@threads for i in 1:m
        ys[i] = log(xs[i])
    end
end


function inv!(xs::Vector{T}, m) where {T}
    Threads.@threads for i in 1:m
        xs[i] = one(T) / xs[i]
    end
end


function log_likelihood(X::SparseMatrixCSC, Xt::SparseMatrixCSC, xs::Vector)
    frag_probs = pAt_mul_B(Xt, X, xs)
    return sum(log.(frag_probs))
end


# assumes a flat prior on π
function log_likelihood(
        frag_probs, log_frag_probs,
        X::SparseMatrixCSC, Xt::SparseMatrixCSC, xs, x_grad,
        ::Val{gradonly}) where {gradonly}
    m, n = size(X)

    # conditional fragment probabilities
    pAt_mul_B!(frag_probs, Xt, xs)

    # log likelihood
    lp = 0.0
    if !gradonly
        log!(log_frag_probs, frag_probs, m)
        lp = sum(log_frag_probs)
        @assert isfinite(lp)
    end

    pAt_mulinv_B!(x_grad, X, frag_probs)

    return lp
end


function factored_log_likelihood(
        frag_probs, log_frag_probs,
        X::SparseMatrixCSC, Xt::SparseMatrixCSC, ks::Vector{Int},
        xs, x_grad,
        ::Val{gradonly}) where {gradonly}
    m, n = size(X)

    # conditional fragment probabilities
    pAt_mul_B!(frag_probs, Xt, xs)

    # log likelihood
    lp = 0.0
    if !gradonly
        log!(log_frag_probs, frag_probs, m)
        log_frag_probs .*= ks
        lp = sum(log_frag_probs)
        @assert isfinite(lp)
    end

    Threads.@threads for i in 1:m
        frag_probs[i] = ks[i] / frag_probs[i]
    end

    pAt_mul_B!(x_grad, X, frag_probs)

    return lp
end


# prior probability correction: without this, there is an implicit
# assumption of a uniform prior over effective length weighted expression
# values. We actually want to assume a uniform prior over unweighted
# expression values, so we correct using the log determinant of the
# jacobian of the effective length transformation.
function effective_length_jacobian_adjustment!(efflens, xs, xls, x_grad)
    n = length(efflens)

    x_scaled_sum = 0.0
    for i in 1:n
        xls[i] = xs[i] / efflens[i]
        x_scaled_sum += xls[i]
    end
    xls ./= x_scaled_sum

    for i in 1:n
        x_grad[i] -= n * (1/efflens[i]) / x_scaled_sum
    end

    # TODO: if I care about the objective function value, I might want to
    # compute that.
    return 0.0
end



function gene_noninformative_prior!(
        efflens, xls, xl_grad, xs, x_grad, gene_transcripts::Dict{String, Vector{Int}})

    # compute gradiens with respect to xl
    fill!(xl_grad, 0.0)
    max_k = 0
    for transcripts in values(gene_transcripts)
        k = length(transcripts)
        max_k = max(max_k, k)
        if k > 1
            c = 0.0
            for i in transcripts
                c += xls[i]
            end

            for i in transcripts
                xl_grad[i] = -(k-1)/c
            end
        end
    end

    # Now compute gradients with respect to x
    n = length(xs)
    x_scaled_sum = 0.0
    for i in 1:n
        x_scaled_sum += xs[i] / efflens[i]
    end
    x_scaled_sum_sq = x_scaled_sum^2

    offdiag_contrib = 0.0
    for i in 1:n
        offdiag_contrib += -xl_grad[i] * xls[i]
    end
    offdiag_contrib /= x_scaled_sum_sq

    for i in 1:length(xs)
        grad_a = xl_grad[i] * ((1/efflens[i]) / x_scaled_sum)
        grad_b = (1/efflens[i]) * offdiag_contrib
        grad = grad_a + grad_b
        x_grad[i] += grad
    end

    # TODO: if I care about the objective function value, I might want to
    # compute that.
    return 0.0
end

