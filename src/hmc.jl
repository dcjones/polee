
using HDF5

const NUM_SAMPLES = 100
const NUM_SUBSAMPLES = 10
const ϵ = 1.0

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
end

function Model(m, n)
    return Model(Int(m), Int(n), Array(Float32, n), Array(Float32, m),
                 Array(Float32, n), Array(Float32, n), Array(Float32, n),
                 Array(Float32, n))
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
function simplex!(xs, grad, xs_sum, zs, zs_log_sum, ys)
    k = length(ys)
    @assert length(xs) == k
    @assert length(grad) == k
    @assert length(xs_sum) == k
    @assert length(zs) == k
    @assert length(zs_log_sum) == k

    ladj = 0.0
    xsum = 0.0
    z_log_sum = 0.0
    xs_sum[1] = 0.0
    zs_log_sum[1] = 0.0
    for i in 1:k
        zs[i] = logistic(ys[i] + log(1/(k - i)))
    end

    for i in 1:k-1
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
        grad[i] += 1 - 2*zs[i] +
            (k - i - 1) * (-1 / (1 - xs_sum[i+1])) *
            zs[i] * (1 - zs[i]) * (1 - xs_sum[i])
    end

    return ladj
end



# assumes a flat prior on π
function log_post(model::Model, X, π, grad)
    frag_probs = model.frag_probs
    fill!(grad, 0.0)

    # transform π to simplex
    ladj = simplex!(model.π_simplex, grad, model.xs_sum, model.zs,
                    model.zs_log_sum, π)

    # conditional fragment probabilities
    A_mul_B!(frag_probs, X, model.π_simplex)
    #@show (minimum(frag_probs), maximum(frag_probs))

    # log-likelihood
    lp = 0.0
    for i in 1:model.m
        lp += log(frag_probs[i])

        # invert for gradient computation
        frag_probs[i] = 1 / frag_probs[i]
    end

    # gradients
    raw_grad = model.raw_grad
    At_mul_B!(raw_grad, X, frag_probs)

    # compute the gradients the correct but intractable way
    zs = model.zs
    zs_log_sum = model.zs_log_sum
    xs_sum = model.xs_sum
    for i in 1:model.n
        for j in i:model.n
            deriv_x_j_y_i = zs[i] * (1 - zs[i])
            deriv_x_j_y_i *= (1 - xs_sum[i])

            if j > i
                deriv_x_j_y_i *=
                    -zs[j] * exp(zs_log_sum[j] - zs_log_sum[i+1])
            end

            grad[i] += raw_grad[j] * deriv_x_j_y_i
        end
    end

    # TODO: figure out how to adjust raw_grad in linear time

    return lp + ladj
end



# TODO: understand what the jacobian term is to handl hypercube -> simple
# transformation


function main()
    input = h5open("output.h5", "r")
    m = read(input["m"]) # number of reads
    n = read(input["n"]) # number of transcripts
    colptr = read(input["colptr"])
    rowval = read(input["rowval"])
    nzval = read(input["nzval"])
    @show (m, n)

    X = SparseMatrixCSC(m, n, colptr, rowval, nzval)
    π = zeros(Float32, n)
    ϕ = zeros(Float32, n)
    grad = Array(Float32, n)

    model = Model(m, n)

    lp0 = log_post(model, X, π, grad)

    @show lp0

    ε = 1e-4
    grad_ = Array(Float32, n)
    for j in 1:n
        πj = π[j]
        π[j] += ε
        lp = log_post(model, X, π, grad_)
        π[j] = πj
        numgrad = (lp - lp0) / ε
        @printf("%d\t%0.4e\t%0.4e\t%0.2f\n", j, grad[j], numgrad,
                (grad[j] - numgrad) / ((grad[j] + numgrad) / 2))
    end

    # check gradient

    #for sample_num in 1:NUM_SAMPLES
        ##ϕ += 0.5 * ϵ * grad_lp0
        #for sum_sample_num in 1:NUM_SUBSAMPLES
            ## TODO
        #end
    #end
end


main()
