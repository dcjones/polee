
using Distributions
using LinearAlgebra

# Gibbs sampler for a gaussian graphical model with graphical horseshoe prior.

struct GHSBlock
    span::UnitRange{Int}
    p::Int # block size
    n::Int # number of samples

    S::Matrix{Float32} # = X'*X (data matrix multiplied by itself)
    si::Vector{Float32}

    Ω::Matrix{Float32} # precision matrix

    Σ::Matrix{Float32} # covariance matrix
    Σi::Matrix{Float32} # covariance matrix ith principal minor
    σi::Vector{Float32} # deleted column from Σ

    Ωinvi::Matrix{Float32} # precision matrix ith principal minor

    Cinv::Matrix{Float32} # inverse of C

    Λ::Matrix{Float32}
    λi::Vector{Float32}

    Ν::Matrix{Float32}
    νi::Vector{Float32}

    a::Vector{Float32} # p-1 length intermedian value
    β::Vector{Float32}
end


# there must be a stdlib function that does this, right?
function ident(p)
    A = zeros(Float32, (p,p))
    for i in 1:p
        A[i,i] = 1.0f0
    end
    return A
end


"""
Construct a block from the span and sample size n.
"""
function GHSBlock(i::Int, j::Int, n::Int)
    p = j-i+1

    return GHSBlock(
        i:j,
        p, n,

        # S
        Array{Float32}(undef, (p, p)),
        Array{Float32}(undef, p-1),

        # Ω
        ident(p),

        # Σ
        ident(p),
        Array{Float32}(undef, (p-1, p-1)),
        Array{Float32}(undef, p-1),

        # Ωinv
        Array{Float32}(undef, (p-1, p-1)),

        # C
        Array{Float32}(undef, (p-1, p-1)),

        # Λ
        ones(Float32, (p, p)),
        Array{Float32}(undef, p-1),

        # Ν
        ones(Float32, (p, p)),
        Array{Float32}(undef, p-1),

        # a, β
        Array{Float32}(undef, p-1),
        Array{Float32}(undef, p-1))
end


function update_S!(block, X)
    Xblock = view(X, :, block.span)
    mul!(block.S, transpose(Xblock), Xblock)
end



"""
Compute ith principal minor (matrix formed by deleting ith row and column)
and store in Xi.

Deleted column/row is stored in xi.
"""
function principal_minor!(Xi::Matrix, xi::Vector,  X::Matrix, i::Int)
    @assert size(Xi, 1) + 1 == size(X, 1)
    @assert size(Xi, 2) + 1 == size(X, 2)

    for u in 1:size(X, 2)
        soff = size(X, 1) * (u-1)

        if u == i
            unsafe_copyto!(xi, 1, X, soff+1, i-1)
            unsafe_copyto!(xi, i, X, soff+i+1, size(X, 1)-1)
            continue
        end

        doff = size(Xi, 1) * (u < i ? u-1 : u-2)
        unsafe_copyto!(Xi, doff+1, X, soff+1, i-1)
        unsafe_copyto!(Xi, doff+i, X, soff+i+1, size(X, 1)-i)
    end
end

# invert a symmetric positive definate matrix without any intermediate allocation
# Both 'src' and 'dest' get overwritten here, the former with junk, the latter with
# the inverse of what was in src.
function inv_inplace!(dest::Matrix, src::Matrix)
    LAPACK.potrf!('U', src)
    LinearAlgebra.inv!(UpperTriangular(src))

    m = size(dest, 1)
    for i in 1:m, j in i:m
        dest[i,j] = 0.0f0
        for k in j:m
            dest[i,j] += src[i,k] * src[j,k]
        end
    end
    for i in 1:m, j in 1:i-1
        dest[i,j] = dest[j,i]
    end
    # Ok, but what if we instead multiplied into the lowert triangle
    # then symmetrized

end


"""
Store a vector a consisting of entries A[i,j] for all j!=i (or A[j,i]), in
a symmetric square matrix A.
"""
function store_rowcol!(A::Matrix, ai::Vector, i::Int)
    p = size(A, 1)
    @assert size(A, 2) == p
    @assert length(ai) == p-1
    for u in 1:i-1
        A[i,u] = ai[u]
        A[u,i] = ai[u]
    end
    for u in i+1:p
        A[i,u] = ai[u-1]
        A[u,i] = ai[u-1]
    end
end


function load_rowcol!(ai::Vector, A::Matrix, i::Int)
    p = size(A, 1)
    @assert size(A, 2) == p
    @assert length(ai) == p-1
    for u in 1:i-1
        ai[u] = A[i,u]
    end
    for u in i+1:p
        ai[u-1] = A[i,u]
    end
end


function symmetrize_from_upper!(A::Matrix)
    for u in 1:size(A, 1), v in 1:u-1
        A[u,v] = A[v,u]
    end
end


function randn!(zs::Vector{T}) where {T}
    for i in 1:length(zs)
        zs[i] = randn(T)
    end
end


function sample!(block::GHSBlock, τ)
    p = block.p
    n = block.n

    S = block.S
    si = block.si

    Σ = block.Σ
    Σi = block.Σi
    σi = block.σi

    Ω = block.Ω

    Ωinvi = block.Ωinvi

    Cinv = block.Cinv

    Λ = block.Λ
    λi = block.λi

    Ν = block.Ν
    νi = block.νi

    a = block.a
    β = block.β

    τ2 = τ^2

    for i in 1:p
        principal_minor!(Σi, σi, Σ, i)
        load_rowcol!(λi, Λ, i)
        load_rowcol!(si, S, i)
        load_rowcol!(νi, Ν, i)

        # sample γ
        γ = rand(Gamma(n/2+1, 2/S[i,i]))

        # sample β
        for u in 1:p-1, v in u:p-1
            Ωinvi[u,v] = Σi[u,v] - σi[u] * σi[v] / Σ[i,i]
        end
        symmetrize_from_upper!(Ωinvi)

        for u in 1:p-1, v in u:p-1
            Cinv[u,v] = S[i,i]*Ωinvi[u,v]
        end
        for u in 1:p-1
            Cinv[u,u] += inv(λi[u] * τ2)
        end
        symmetrize_from_upper!(Cinv)

        Cinv_chol = cholesky!(Cinv)
        randn!(β)
        ldiv!(Cinv_chol, β)

        ldiv!(Cinv_chol, si)
        neg_μ_β = si
        β .-= neg_μ_β

        # compute Ω[i,i]
        mul!(a, Ωinvi, β)
        Ω[i,i] = γ
        # add β'*Ωi*β
        for u in 1:p-1
            Ω[i,i] += β[u]*a[u]
        end

        # update Ω[i,:] and Ω[:,i]
        store_rowcol!(Ω, β, i)

        # sample λ and ν
        for u in 1:p-1
            h = u < i ? u : u + 1
            scale = 1.0f0/νi[u] + Ω[i,h]^2/(2*τ^2)
            λi[u] = rand(InverseGamma(1.0, scale))

            # TODO: these local shrinkage parameters can get too small
            # causing things to break. We should threshold them somehow.

            scale = 1.0f0 + 1.0f0/λi[u]
            νi[u] = rand(InverseGamma(1.0, scale))
        end

        # update Σ
        for u in 1:p, v in u:p
            if u == i && v != i
                k = v < i ? v : v-1
                Σ[u,v] = -a[k]/γ
            elseif u != i && v == i
                h = u < i ? u : u-1
                Σ[u,v] = -a[h]/γ
            elseif u != i && v != i
                h = u < i ? u : u-1
                k = v < i ? v : v-1
                Σ[u,v] = Ωinvi[h,k] + a[h]*a[k]/γ
            end
        end
        Σ[i,i] = 1.0f0/γ
        symmetrize_from_upper!(Σ)

        store_rowcol!(Λ, λi, i)
        store_rowcol!(Ν, νi, i)
    end

    # TODO: this has to happen outside all the blocks

    # sample τ
    # TODO:

    # sample ξ
    # TODO:
end



Ωtrue = [
    20.0 -15.0 0.0 0.0
    -15.0 20.0 0.0 0.0
    0.0 0.0 20.0 0.0
    0.0 0.0 0.0 20.0 ]

# Ωtrue = ident(4)

Σtrue = inv(Ωtrue)

n = 50
X = transpose(rand(MultivariateNormal(Σtrue), n))

block = GHSBlock(1, size(X, 2), n)
update_S!(block, X)
@show size(block.S)

for _ in 1:1000
    sample!(block, 1.0f0)
    @show block.Ω
    @show block.Λ
end


