

immutable Kumaraswamy{T<:Real} <: Distributions.ContinuousUnivariateDistribution
    a::T
    b::T
end


function Base.rand(d::Kumaraswamy)
    p = rand()
    return (1 - (1 - p)^(1/d.b))^(1/d.a)
end


function harmonic(x)
    return digamma(x+1) + γ
end


function Distributions.entropy(d::Kumaraswamy)
    return (1 - 1/d.a) + (1 - 1/d.b) * harmonic(b) + log(d.a * d.b)
end



immutable KumaraswamyTransform

end

# Take uniform random numbers in zs, transform them to Kamaraswamy distributed values
# in ys according to parameters as and bs. Return the log absolute determinant of the jacobian,
function kumaraswamy_transform!(as::Vector{Float32}, bs::Vector{Float32},
                                zs::Vector{Float32}, ys::Vector{Float32})
    @assert length(as) == length(bs) == length(zs) == length(ys)
    ladj = 0.0f0
    for i in 1:length(ys)
        a = as[i]
        b = bs[i]
        z = zs[i]
        ia = 1/a
        ib = 1/b
        c = 1 - (1 - z)^ib
        ys[i] = c^ia
        ladj += (ib - 1) * log(1 - z) + (ia - 1) * log(c) - log(a * b)
    end

    return ladj
end


function kumaraswamy_transform_gradients!(as, bs, y_grad, a_grad, b_grad)
    for i in 1:length(ys)
        a = as[i]
        b = bs[i]
        z = zs[i]
        ia = 1/a
        ib = 1/b
        c = 1 - (1 - z)^ib
        log_omz = log(1 - z)

        # derivative of log jacobian determinant
        a_grad[i] += -log(c) / a^2 - ia
        b_grad[i] += -log_omz / b^2 +
                      (ia - 1) * (1/c) * (1 - z)^ib * log_omz / b^2 -
                      ib

        # df/da and df/db, computed as dy/da * df/dy and dy/db * df/dy
        dy_da = -c^ia * log(c) / a^2
        a_grad[i] += dy_da * y_grad[i]

        dy_db = c^(ia - 1) * (1 - z)^ib * log(1 - z) / (a * b^2)
        b_grad[i] += dy_db * y_grad[i]
    end
end

