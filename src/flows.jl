

immutable HouseholderFlow
    A::Matrix{Float32} # n-by-k matrix representing k householder transformations
    A_grad::Matrix{Float32}
    norm::Vector{Float32}    # ||v||^2, computed by transform!
    c::Vector{Float32}       # 2v^Tx // ||v||^2, computed by transform!
end


function HouseholderFlow(n, k)
    A = randn(Float32, (n, k))
    return HouseholderFlow(A, similar(A), Array{Float32}(k), Array{Float32}(k))
end


function transform!(t::HouseholderFlow, x)
    A = t.A
    n, k = size(A)
    @assert length(x) >= n

    println("Flow")
    @show minimum(x), maximum(x)

    for j in 1:k
        c = 0.0
        norm = 0.0
        for i in 1:n
            norm += A[i,j]^2
            c += A[i,j] * x[i]
        end
        c *= 2
        c /= norm

        for i in 1:n
            # if i < 10
            #     @show (i, x[i], A[i,j], c, A[i,j]*c)
            # end

            x[i] -= A[i,j] * c
        end
        t.norm[j] = norm
        t.c[j] = c
    end

    @show minimum(x), maximum(x)

    return x
end


function gradients!(t::HouseholderFlow, x, x_grad)
    A = t.A
    A_grad = t.A_grad
    n, k = size(A)
    @assert size(A_grad) == (n, k)
    @assert length(x) >= n
    @assert length(x_grad) >= n

    x_grad_sum = 0.0
    for i in 1:n
        x_grad_sum += x_grad[i]
    end

    for j in k:-1:1
        norm = t.norm[j]
        c = t.c[j]

        weighted_x_grad_sum = 0.0
        for i in 1:n
            weighted_x_grad_sum += A[i,j] * x_grad[i]
        end

        # update x gradients
        for i in 1:n
            x_grad[i] -= (2 * A[i,j] / norm) * weighted_x_grad_sum
        end

        # update A gradients
        for i in 1:n
            A_grad[i,j] +=
                -x_grad[i] * c -
                (2 * x[i] / norm) * weighted_x_grad_sum +
                (A[i,j] * c / norm) * x_grad_sum
        end
    end
end
