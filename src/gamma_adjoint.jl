

"""
This is essentially gamma_inc(p, x)[1] from SpecialFunctions, but amenable to AD.

Note that this can run into numerical issues if T != Float64.
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
    tol = T(1.0e-8)
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

# f32rand(d::Beta{Float32}) = f32rand(Random.GLOBAL_RNG, d)

# function f32rand(rng::AbstractRNG, d::Beta{Float32})
#     (α, β) = params(d)
#     g1 = f32rand(rng, Gamma(α, one(Float32)))
#     g2 = f32rand(rng, Gamma(β, one(Float32)))
#     return g1 / (g1 + g2)
# end


# function rand_beta(rng::AbstractRNG, α::Float32, β::Float32)
#     g1 = f32rand(rng, Gamma(α, one(Float32)))
#     g2 = f32rand(rng, Gamma(β, one(Float32)))
#     return g1 / (g1 + g2)
# end

# # TODO: try to speed things up further by adapting the gamma code for f32
# f32rand(rng::AbstractRNG, d::Gamma{Float32}) = Float32(rand(rng, d))


# ZygoteRules.@adjoint function f32rand(rng::AbstractRNG, d::Gamma{T}) where {T<:Real}
#     z = f32rand(rng, d)
#     function rand_gamma_pullback(c)
#         y = z/d.θ
#         ∂α, ∂y = gradient(αy -> Zygote.forwarddiff(_gamma_inc_lower, αy), SA[d.α, y])[1]
#         return (
#             nothing,
#             (α=(-d.θ*∂α/∂y)*c,
#              θ=y*c))
#     end
#     return z, rand_gamma_pullback
# end


# TODO: This method is unstable for small values of α and β
# We maybe have to use the full version.

function rand_betas(αs::Vector, βs::Vector)
    us = rand_gamma1s(αs)
    vs = rand_gamma1s(βs)
    eps = 1e-16

    us = clamp.(us, eps, 1 - eps)
    vs = clamp.(vs, eps, 1 - eps)
    return us ./ (us .+ vs)
end


function rand_gamma1s(αs::Vector)
    n = length(αs)
    us = Vector{Float64}(undef, n)
    Threads.@threads for i in 1:n
        us[i] = rand_gamma1(αs[i])
    end
    return us
end


function rand_gamma1(α)
    rand(Gamma(α, 1.0))
end


ZygoteRules.@adjoint function rand_gamma1s(αs::Vector)
    xs = rand_gamma1s(αs)

    function rand_gamma1s_pullback(x̄)
        n = length(xs)
        ∂αs = Vector{Float32}(undef, n)
        Threads.@threads for i in 1:n
            dα, dy = gradient(
                αy -> Zygote.forwarddiff(_gamma_inc_lower, αy),
                SA[Float64(αs[i]), Float64(xs[i])])[1]
            ∂αs[i] = iszero(x̄[i]) ? zero(Float32) : -dα/dy*x̄[i]
            if !isfinite(∂αs[i])
                @show (∂αs[i], αs[i], dα, dy, x̄[i], xs[i])
                error()
            end
        end
        return (∂αs,)
    end

    return xs, rand_gamma1s_pullback
end


# TODO: Putting this here because I don't have a better place. We'll need to
# reorganize once we figure some things out.


"""
Pathwise derivative of a random variable z ~ Gamma(α, 1), wrt to α.
"""
function gamma1_grad(z, α)
    _, ∂αz = Zygote.forward_jacobian(_gamma_inc_lower, SA[Float64(α), Float64(z)])
    return -∂αz[1]/∂αz[2]
end


"""
Beta distribution entropy.
"""
function beta_entropy(α, β)
    return entropy(Beta(α, β))
end


"""
Gradient of beta distribution entropy.
"""
function beta_entropy_grad(α, β)
    c = (α + β - 2) * trigamma(α + β)
    ∂α = c - (α - 1) * trigamma(α)
    ∂β = c - (β - 1) * trigamma(β)
    return (∂α, ∂β)
end

