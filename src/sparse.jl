

"""
Multithreaded version of At_mul_B!
"""
function pAt_mul_B!{T,I}(y::Vector{T}, A::SparseMatrixCSC{T,I}, x::Vector{T})
    fill!(y, zero(T))
    Threads.@threads for j in 1:A.n
        @inbounds for k in A.colptr[j]:A.colptr[j+1]-1
            i = A.rowval[k]
            y[j] += x[i] * A.nzval[k]
        end
    end
end


