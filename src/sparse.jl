
# TODO: If https://github.com/JuliaLang/julia/pull/29525 every gets merged,
# we can just use that. In the mean time...

"""
Multithreaded version of At_mul_B!
"""
function pAt_mul_B!(y::Vector{S}, A::SparseMatrixCSC{T,I}, x::Vector) where {S,T,I}
    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval

    Threads.@threads for j in 1:length(y)
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
