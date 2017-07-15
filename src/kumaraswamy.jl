

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


function harmonic_deriv(x)
    return trigamma(x+1)
end


function Distributions.entropy(d::Kumaraswamy)
    return (1 - 1/d.a) + (1 - 1/d.b) * harmonic(b) + log(d.a * d.b)
end


# Take uniform random numbers in zs, transform them to Kamaraswamy distributed values
# in ys according to parameters as and bs. Return the log absolute determinant of the jacobian,
function kumaraswamy_transform!(as::Vector, bs::Vector,
                                zs::Vector, ys::Vector,
                                work::Vector)
    Threads.@threads for i in 1:length(ys)
        a = Float64(as[i])
        b = Float64(bs[i])
        z = Float64(zs[i])
        ia = 1/a
        ib = 1/b

        # u = exp(ib * log1p(-z)) # (1-z)^(1/b)
        # c = 1 - u

        c = 1 - (1 - z)^ib
        ys[i] = c^ia
        ys[i] = min(1.0f0 - eps(Float32), max(eps(Float32), ys[i]))

        # ladj term
        work[i] = (ib - 1) * log(1 - z) + (ia - 1) * log(c) - log(a * b)
    end

    ladj = sum(work)
    @assert isfinite(ladj)
    return ladj
end


function kumaraswamy_transform_gradients!(zs, as, bs, y_grad, a_grad, b_grad)
    Threads.@threads for i in 1:length(as)
        a = Float64(as[i])
        b = Float64(bs[i])
        z = Float64(zs[i])
        ia = 1/a
        ib = 1/b
        c = 1 - (1 - z)^ib
        log_c = log(c)
        log_omz = log(1 - z)

        # derivative of log jacobian determinant
        a_grad[i] += -log_c / a^2 - ia
        b_grad[i] += -log_omz / b^2 +
                      (ia - 1) * (1/c) * (1 - z)^ib * log_omz / b^2 -
                      ib

        # df/da and df/db, computed as dy/da * df/dy and dy/db * df/dy
        dy_da = -c^ia * log_c / a^2
        a_grad[i] += dy_da * y_grad[i]

        dy_db = c^(ia - 1) * (1 - z)^ib * log_omz / (a * b^2)
        b_grad[i] += dy_db * y_grad[i]
    end
end

