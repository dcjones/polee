

"""
This is essentially gamma_inc(p, x)[1] from SpecialFunctions, but amenable to AD.
"""
function _gamma_inc_lower(px::AbstractVector{T}) where {T<:Real}
    return _gamma_inc_lower(px[1], px[2])
end

function _gamma_inc_lower(p::T, x::T) where {T<:Real}
    if p <= zero(T)
        throw(DomainError(p, "p > 0 required for gamma_inc"))
    end

    elimit = T(-88.0)
    oflo = T(1.0e37)
    plimit = T(1000.0)
    tol = T(1.0e-14)
    xbig = T(1.0e8)

    if x < zero(T)
        throw(DomainError(x, "x >= 0 required for gamma_inc"))
    end

    if p <= zero(T)
        throw(DomainError(x, "p > 0 required for gamma_inc"))
    end

    if x == zero(T)
        return zero(T)
    end

    # Use a normal approximation for large p
    if plimit < p
        pn1 = 3*sqrt(p) * ((x/p)^(one(T)/3) + one(T) / (9*p) - one(T))
        return Distributions._normcdf(pn1)
    end

    # If x is extremely large return 1
    if xbig < x
        return one(T)
    end

    # Use Pearson's series expansion
    if x <= one(T) || x < p
        value = zero(T)
        arg = p*log(x) - x - NaNMath.lgamma(p + one(T))
        c = one(T)
        value = one(T)
        a = p

        while true
            a += one(T)
            c *= x/a
            value += c
            if c <= tol
                break
            end
        end

        arg += log(value)
        return arg < elimit ? zero(T) : exp(arg)

    # Use a continued fraction expansion
    else
        arg = p*log(x) - x - NaNMath.lgamma(p)
        a = one(T) - p
        b = a + x + one(T)
        c = zero(T)
        pn1 = one(T)
        pn2 = x
        pn3 = x + one(T)
        pn4 = x*b
        value = pn3 / pn4

        while true
            a += one(T)
            b += 2*one(T)
            c += one(T)
            an = a*c
            pn5 = b*pn3 - an*pn1
            pn6 = b*pn4 - an*pn2
            if abs(pn6) > zero(T)
                rn = pn5/pn6
                if abs(value - rn) <= min(tol, tol*rn)
                    break
                end
                value = rn
            end

            pn1 = pn3
            pn2 = pn4
            pn3 = pn5
            pn4 = pn6
            if abs(pn5) >= oflo
                # re-scale terms in continued fraction if terms are large
                pn1 /= oflo
                pn2 /= oflo
                pn3 /= oflo
                pn4 /= oflo
            end
        end

        arg += log(value)
        return arg < elimit ? one(T) : one(T) - exp(arg)
    end
end


ZygoteRules.@adjoint function Distributions.rand(rng::AbstractRNG, d::Gamma{T}) where {T<:Real}
    z = rand(rng, d)
    function rand_gamma_pullback(c)
        y = z/d.θ
        ∂α, ∂y = gradient(αy -> Zygote.forwarddiff(_gamma_inc_lower, αy), SA[d.α, y])[1]
        return (
            nothing,
            (α=(-d.θ*∂α/∂y)*c,
             θ=y*c))
    end
    return z, rand_gamma_pullback
end



# This code is taken from Distributions but adapted to generate Float32
# rands to hopefuling speed things up.

f32rand(d::Beta{Float32}) = f32rand(Random.GLOBAL_RNG, d)

function f32rand(rng::AbstractRNG, d::Beta{Float32})
    (α, β) = params(d)
    g1 = f32rand(rng, Gamma(α, one(Float32)))
    g2 = f32rand(rng, Gamma(β, one(Float32)))
    return g1 / (g1 + g2)
end


function rand_beta(rng::AbstractRNG, α::Float32, β::Float32)
    g1 = f32rand(rng, Gamma(α, one(Float32)))
    g2 = f32rand(rng, Gamma(β, one(Float32)))
    return g1 / (g1 + g2)
end

# TODO: try to speed things up further by adapting the gamma code for f32
f32rand(rng::AbstractRNG, d::Gamma{Float32}) = Float32(rand(rng, d))


ZygoteRules.@adjoint function f32rand(rng::AbstractRNG, d::Gamma{T}) where {T<:Real}
    z = f32rand(rng, d)
    function rand_gamma_pullback(c)
        y = z/d.θ
        ∂α, ∂y = gradient(αy -> Zygote.forwarddiff(_gamma_inc_lower, αy), SA[d.α, y])[1]
        return (
            nothing,
            (α=(-d.θ*∂α/∂y)*c,
             θ=y*c))
    end
    return z, rand_gamma_pullback
end
