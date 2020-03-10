

struct Kumaraswamy{T<:Real}
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


# Take uniform random numbers in zs, transform them to Kamaraswamy distributed values
# in ys according to parameters as and bs. Return the log absolute determinant of the jacobian,
function kumaraswamy_transform!(
        as::Vector, bs::Vector, zs::Vector, ys::Vector, work::Vector,::Val{compute_ladj}) where {compute_ladj}
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

        # ladj term
        if compute_ladj
            work[i] = (ib - 1) * log(1 - z) + (ia - 1) * log(c) - log(a * b)
        end
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


function kumaraswamy_median(a, b)
    return (1 - 2^(-1/b))^(1/a)
end


function kumaraswamy_median_grad(a, b)
    c = (1 - 2^(-1/b))
    med = c^(1/a)
    df_da = -med * log(c) / a^2
    df_db = -(2^(-1/b) * log(2) * c^(1/a - 1)) / (a * b^2)
    return (med, df_da, df_db)
end


function kumaraswamy_moment(a, b, mn)
    return b * beta(1 + mn/a, b)
end


function kumaraswamy_moment_grad(a, b, mn)

    beta_value = beta(1 + mn/a, b)
    digamma_value1 = digamma(1 + mn/a + b)

    f = b * beta_value

    df_da = - b * mn * beta_value * (digamma(1 + mn/a) - digamma_value1) / a^2
    df_db = beta_value * (1 + b * (digamma(b) - digamma_value1))
    return f, df_da, df_db
end


function kumaraswamy_fit_moments(mean, var)
    m1 = mean
    m2 = var + mean^2

    @show (mean, var, m1, m2)

    # ab = Float64[log(10.0), log(921.7)]
    ab = Float64[0.0, 0.0]
    J = Array{Float64}(2, 2)
    f = Array{Float64}(2)
    for _ in 1:20
        a, b = exp(ab[1]), exp(ab[2])
        f[1] = kumaraswamy_moment(a, b, 1) - m1
        f[2] = kumaraswamy_moment(a, b, 2) - m2

        f[1], J[1,1], J[1,2] = kumaraswamy_moment_grad(a, b, 1)
        f[2], J[2,1], J[2,2] = kumaraswamy_moment_grad(a, b, 2)

        # adjust for values being fit
        f[1] -= m1
        f[2] -= m2

        if max(abs(f[1]), abs(f[2])) < 1e-7
            break
        end

        # account for log transform
        J[1,1] *= a
        J[2,1] *= a
        J[1,2] *= b
        J[2,2] *= b

        @show ab
        @show f
        @show J
        @show inv(J)

        ab .-= inv(J) * f
    end

    return ab[1], ab[2]
end


function kumaraswamy_fit_median_var(med, var)
    ab = Float64[0.0, 0.0]
    J = Array{Float64}(undef, (2, 2))
    f = Array{Float64}(undef, 2)

    ab[1] = 1.0
    ab[2] = 1.0

    for _ in 1:20
        a, b = exp(ab[1]), exp(ab[2])

        # f[1], J[1,1], J[1,2] = kumaraswamy_moment_grad(a, b, 1)
        f[1], J[1,1], J[1,2] = kumaraswamy_median_grad(a, b)

        mean, dmean_da, dmean_db = kumaraswamy_moment_grad(a, b, 1)
        f[2], J[2,1], J[2,2] = kumaraswamy_moment_grad(a, b, 2)

        # center variance around mean
        f[2] -= mean^2
        J[2,1] -= 2*mean * dmean_da
        J[2,2] -= 2*mean * dmean_db

        # center variance around median
        # f[2] = f[2] - f[1]^2
        # J[2,1] -= 2*f[1] * J[1,1]
        # J[2,2] -= 2*f[1] * J[1,2]

        # adjust for values being fit
        f[1] -= med
        f[2] -= var

        if max(abs(f[1]), abs(f[2])) < 1e-7
            break
        end

        # account for log transform
        J[1,1] *= a
        J[2,1] *= a
        J[1,2] *= b
        J[2,2] *= b

        # @show ab
        # @show J
        # @show f
        Jinv = inv(J)
        delta = Jinv * f
        # @show delta

        max_b = 15

        if ab[2] >= max_b && delta[2] < 0
            # just update a
            ab[1] -= f[1] / J[1,1]

            J[1,2] = 0.0
            J[2,2] = 0.0

            j = J[:,1]
            jt = transpose(j)
            jinv = inv(jt * j) * jt

            ab[1] -= jinv * f
        else
            if ab[2] - delta[2] > max_b
                c = (ab[2] - max_b) / delta[2]
                ab .-= c .* delta
            else
                ab .-= delta
            end
        end
    end

    return ab[1], ab[2]
end

