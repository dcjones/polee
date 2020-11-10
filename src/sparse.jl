
# TODO: If https://github.com/JuliaLang/julia/pull/29525 every gets merged,
# we can just use that. In the mean time...

"""
Multithreaded version of At_mul_B!
"""
function pAt_mul_B!(y::Vector{S}, A::SparseMatrixCSC{T,I}, x::AbstractVector) where {S,T,I}
    @assert length(x) == size(A, 1)
    @assert length(y) == size(A, 2)
    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval

    # Threads.@threads for j in 1:length(y)
    for j in 1:length(y)
        @inbounds begin
            tmp = zero(T)
            for k_ in colptr[j]:colptr[j+1]-1
                k = Int(k_)
                i = Int(rowval[k])
                tmp += x[i] * nzval[k]
            end
            y[j] = tmp
        end
    end
end


# multiply by 1./x instead of x
function pAt_mulinv_B!(y::Vector{S}, A::SparseMatrixCSC{T,I}, x::Vector) where {S,T,I}
    @assert length(x) == size(A, 1)
    @assert length(y) == size(A, 2)
    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval

    Threads.@threads for j in 1:length(y)
        @inbounds begin
            tmp = zero(T)
            for k_ in colptr[j]:colptr[j+1]-1
                k = Int(k_)
                i = Int(rowval[k])
                tmp += nzval[k] / x[i]
            end
            y[j] = tmp
        end
    end
end


function pAt_mul_B(A::SparseMatrixCSC, At::SparseMatrixCSC, x::AbstractVector{T}) where {T}
    y = Vector{T}(undef, size(A, 2))
    pAt_mul_B!(y, A, x)
    return y
end


ZygoteRules.@adjoint function pAt_mul_B(A::SparseMatrixCSC, At::SparseMatrixCSC, x::Vector{T}) where {T}
    y = pAt_mul_B(A, At, x)
    function pullback(ȳ)
        return nothing, nothing, pAt_mul_B(At, A, ȳ)
    end

    return y, pullback
end

# function ChainRules.rrule(::typeof(pAt_mul_B), A::SparseMatrixCSC, At::SparseMatrixCSC, x::Vector)
#     println("HERE")
#     y = pAt_mul_B(A, At, x)

#     function pullback(ȳ)
#         return NO_FIELDS, Zero(), Zero(), pAt_mul_B(At, A, ȳ)
#     end

#     return y, pullback
# end
