

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



immutable MultiplicativeMixingFlow
    ws::Vector{Float32}
    xs::Vector{Float32} # input x, prior to transformation
    ws_grad::Vector{Float32}
end


function MultiplicativeMixingFlow(n)
    return MultiplicativeMixingFlow(#randn(Float32, n-1),
                                    zeros(Float32, n-1),
                                    Array{Float32}(n),
                                    Array{Float32}(n-1))
end


function transform!(t::MultiplicativeMixingFlow, xs)
    n = length(t.xs)
    for i in 1:n
        t.xs[i] = xs[i]
    end
    ladj = 0.0
    for i in 2:n
        xs[i] = t.xs[i] * (1.0f0 + t.ws[i-1] * t.xs[i-1])
        ladj += log(abs(1.0f0 + t.ws[i-1] * t.xs[i-1]))
    end

    return ladj
end


function gradients!(t::MultiplicativeMixingFlow, xs, xs_grad)
    n = length(t.xs)
    for i in 1:n-1
        t.ws_grad[i] += t.xs[i] * t.xs[i+1] * xs_grad[i+1]
    end

    if n > 1
        xs_grad[1] = xs_grad[1] + t.ws[1] * t.xs[2] * xs_grad[2]
        for i in 2:n-1
            # if i == 2
            #     @show t.ws[i-1]
            #     @show t.ws[i]
            #     @show (1.0f0 + t.ws[i-1] * t.xs[i-1]) * xs_grad[i]
            #     @show t.ws[i] * t.xs[i] * xs_grad[i+1]
            # end
            xs_grad[i] = (1.0f0 + t.ws[i-1] * t.xs[i-1]) * xs_grad[i] +
                         t.ws[i] * t.xs[i] * xs_grad[i+1]
        end
        xs_grad[n] = (1.0f0 + t.ws[n-1] * t.xs[n-1]) * xs_grad[n]
    end

    # jacobian gradients wrt to x and w
    for i in 1:n-1
        jt = 1.f0 + t.ws[i] * t.xs[i]
        jd = (1.0f0/abs(jt)) * (jt/abs(jt))

        xs_grad[i] += jd * t.ws[i] / (1.0f0 + t.ws[i] * t.xs[i])
        t.ws_grad[i] += jd * t.xs[i] / (1.0f0 + t.ws[i] * t.xs[i])
    end
end


immutable AdditiveMixingFlow
    ws::Vector{Float32}
    xs::Vector{Float32} # input x, prior to transformation
    ws_grad::Vector{Float32}
end


function AdditiveMixingFlow(n)
    return AdditiveMixingFlow(#randn(Float32, n-1),
                              zeros(Float32, n-1),
                              Array{Float32}(n),
                              Array{Float32}(n-1))
end


function transform!(t::AdditiveMixingFlow, xs)
    n = length(t.xs)
    for i in 1:n
        t.xs[i] = xs[i]
    end
    ladj = 0.0
    for i in 2:n
        xs[i] = t.xs[i] + t.ws[i-1] * t.xs[i-1]
    end

    return 0.0
end


function gradients!(t::AdditiveMixingFlow, xs, xs_grad)
    n = length(t.xs)
    for i in 1:n-1
        t.ws_grad[i] += t.xs[i+1] * xs_grad[i+1]
    end

    if n > 1
        xs_grad[1] = xs_grad[1] + t.ws[1] * xs_grad[2]
        for i in 2:n-1
            xs_grad[i] = xs_grad[i] +
                         t.ws[i] * xs_grad[i+1]
        end
        # xs_grad[n] = (1.0f0 + t.ws[n-1] * t.xs[n-1]) * xs_grad[n]
    end
end