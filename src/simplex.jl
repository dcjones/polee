
using Distributions

logistic(x) = 1 / (1 + exp(-x))


"""
Transform an unconstrained vector `ys` to a simplex. Store in `xs`. Compute and
store gradient of log T

"""
function simplex!(xs, grad, ys)
    k = length(ys)
    @assert length(xs) == k

    # TODO: store this externally
    zs = Array(eltype(xs), k)

    # TODO: do we still need this?
    zs_log_sum = Array(eltype(xs), k)

    xs_sum = Array(eltype(xs), k)

    # log absolute determinate of the jacobian
    ladj = 0.0
    xsum = 0.0
    z_log_sum = 0.0
    xs_sum[1] = 0.0
    zs_log_sum[1] = 0.0
    for i in 1:k-1
        zs[i] = logistic(ys[i] + log(1/(k - i)))

        ladj += log(zs[i]) + log(1 - zs[i]) + log(1 - xsum)

        xs[i] = (1 - xsum) * zs[i]

        xsum += xs[i]
        xs_sum[i+1] = xsum

        z_log_sum += log(1 - zs[i])
        zs_log_sum[i+1] = z_log_sum
    end
    xs[k] = 1 - xsum


    # TODO: make sure we can compute derivatives of x correctly
    #l = 8
    #ladj = xs[l]
    #for i in 1:k-1
        #if i > l
            #grad[i] = 0.0
            #continue
        #end

        ## partial of x_l by y_i
        #deriv_x_l_y_i = zs[i] * (1 - zs[i])
        #deriv_x_l_y_i *= (1 - xs_sum[i])

        #if l > i
            #deriv_x_l_y_i *=
                #exp(zs_log_sum[l] - zs_log_sum[i+1])
        #end

        #if l != i
            #deriv_x_l_y_i *= -zs[l]
        #end
        #grad[i] = deriv_x_l_y_i
    #end


    # TODO: make sure this actually works for non constant values of y
    for i in 1:k-1
        grad[i] += 1 - 2*zs[i] +
            (k - i - 1) * (-1 / (1 - xs_sum[i+1])) *
            zs[i] * (1 - zs[i]) * (1 - xs_sum[i])
    end

    return ladj
end



function check_simplex_gradient()
    rng = srand(1234)
    n = 36
    xs = Array(Float64, n)
    grad = zeros(Float64, n)
    grad_ = Array(Float64, n)
    numgrad = Array(Float64, n)
    ϵ = 1e-9
    #ys = scale * rand(rng, n)
    ys = rand(Normal(0, 1), n)
    #ys = zeros(Float32, n)
    #@show ys
    ladj = simplex!(xs, grad, ys)
    @show ladj
    #@show ladj
    #@show xs

    @show ys
    for j in 1:n
        y = ys[j]
        ys[j] += ϵ
        ladj2 = simplex!(xs, grad_, ys)
        numgrad[j] = (ladj2 - ladj) / ϵ
        ys[j] = y
        @printf("%0.4e\t%0.4e\t%0.4e\n", grad[j],
                numgrad[j], grad[j] - numgrad[j])
    end
end


check_simplex_gradient()
