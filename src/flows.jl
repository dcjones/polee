

immutable HouseholderFlow
    A::Matrix{Float32} # n-by-k matrix representing k householder transformations
    A_grad::Matrix{Float32}
    xs::Matrix{Float32}      # intermediate results
    norm::Vector{Float32}    # ||v||^2, computed by transform!
    c::Vector{Float32}       # 2v^Tx // ||v||^2, computed by transform!
end


function HouseholderFlow(n, k)
    A = randn(Float32, (n, k))
    # A = ones(Float32, (n, k))
    # A = zeros(Float32, (n, k))
    # for j in 1:k
    #     A[rand(1:n),j] = 1
    # end
    return HouseholderFlow(A, similar(A), similar(A),
                           Array{Float32}(k), Array{Float32}(k))
end


function transform!(t::HouseholderFlow, x)
    A = t.A
    n, k = size(A)
    @assert length(x) >= n

    for j in 1:k
        for i in 1:n
            t.xs[i,j] = x[i]
        end

        c = 0.0
        norm = 0.0
        for i in 1:n
            norm += A[i,j]^2
            c += A[i,j] * x[i]
        end
        c *= 2
        c /= norm

        for i in 1:n
            x[i] -= A[i,j] * c
        end
        t.norm[j] = norm
        t.c[j] = c
    end

    return x
end


function gradients!(t::HouseholderFlow, x, x_grad)
    A = t.A
    A_grad = t.A_grad
    n, k = size(A)
    @assert size(A_grad) == (n, k)
    @assert length(x) >= n
    @assert length(x_grad) >= n

    for j in k:-1:1
        norm = t.norm[j]
        c = t.c[j]

        # update A gradients

        weighted_x_grad_sum = 0.0
        for i in 1:n
            weighted_x_grad_sum += A[i,j] * x_grad[i]
        end

        for i in 1:n
            A_grad[i,j] +=
                -c * x_grad[i] +
                (2.0/norm) * (A[i,j] * c - t.xs[i,j]) * weighted_x_grad_sum
        end

        # update x gradients

        for i in 1:n
            x_grad[i] -= (2 * A[i,j] / norm) * weighted_x_grad_sum
        end
    end
end
