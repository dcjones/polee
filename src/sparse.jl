

"""
Multithreaded version of At_mul_B!
"""
function pAt_mul_B!{T,I}(y::Vector, A::SparseMatrixCSC{T,I}, x::Vector)
    fill!(y, zero(T))
    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval
    Threads.@threads for j in 1:A.n
        @inbounds for k in colptr[j]:colptr[j+1]-1
            i = rowval[k]
            y[j] += x[i] * nzval[k]
        end
    end
end


