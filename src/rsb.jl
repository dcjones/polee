
module RSB

export RSBMatrix

const librsb = "/usr/local/lib/librsb.so"
const blas_invalid_handle = 261
const blas_no_trans   = 111
const blas_trans      = 112
const blas_conj_trans = 113


typealias BLASSparseMatrix Cint
typealias RSBErr Cint


function rsb_init()
    # init library with default options
    err = ccall((:rsb_lib_init, librsb), RSBErr,
                (Ptr{Void},), C_NULL)

    if err != 0
        error("rsb_lib_init failed with code $(err)")
    end
end


type RSBMatrix
    ptr::BLASSparseMatrix
end


function destroy(M::RSBMatrix)
    err = ccall((:BLAS_usds, librsb), Cint, (BLASSparseMatrix,),
                M.ptr)
    if err != 0
        error("BLAS_usds call failed with code $(err)")
    end
end


function RSBMatrix(A::SparseMatrixCSC)
    m, n = size(A)
    nnz = length(A.nzval)
    indx = Array(Cint, nnz)
    jndx = Array(Cint, nnz)

    k = 1
    for j in 1:n
        while k < A.colptr[j+1]
            i = A.rowval[k]
            indx[k] = i - 1
            jndx[k] = j - 1
            k += 1
        end
    end
    @assert k == nnz + 1

    ptr = ccall((:BLAS_suscr_begin, librsb), BLASSparseMatrix, (Cint, Cint), m, n)
    if ptr == blas_invalid_handle
        error("BLAS_suscr_begin failed")
    end

    # params: A, nnz, val, indx, jndx
    err = ccall((:BLAS_suscr_insert_entries, librsb), RSBErr,
                (BLASSparseMatrix, Cint, Ptr{Float32}, Ptr{Cint}, Ptr{Cint}),
                ptr, nnz, A.nzval, indx, jndx)

    if err != 0
        error("BLAS_suscr_insert_entries failed with code $(err)")
    end

    err = ccall((:BLAS_suscr_end, librsb), Cint, (BLASSparseMatrix,), ptr)
    if err != 0
        error("BLAS_suscr_end failed with code $(err)")
    end

    ret = RSBMatrix(ptr)
    finalizer(ret, destroy)
    return ret
end


function Base.A_mul_B!(y::Vector{Float32}, A::RSBMatrix, x::Vector{Float32})
    fill!(y, 0.0f0)
    err = ccall((:BLAS_susmv, librsb), Cint,
                (Cint,             # transA
                 Float32,          # alpha
                 BLASSparseMatrix, # A
                 Ptr{Float32},     # x
                 Cint,             # incx
                 Ptr{Float32},     # y
                 Cint),            # incy
                blas_no_trans, 1.0f0, A.ptr, x, 1, y, 1)
    if err != 0
        error("BLAS_susmv failed with code $(err)")
    end
    return y
end


function Base.At_mul_B!(y::Vector{Float32}, A::RSBMatrix, x::Vector{Float32})
    fill!(y, 0.0f0)
    err = ccall((:BLAS_susmv, librsb), Cint,
                (Cint,             # transA
                 Float32,          # alpha
                 BLASSparseMatrix, # A
                 Ptr{Float32},     # x
                 Cint,             # incx
                 Ptr{Float32},     # y
                 Cint),            # incy
                blas_trans, 1.0f0, A.ptr, x, 1, y, 1)
    if err != 0
        error("BLAS_susmv failed with code $(err)")
    end
    return y
end


end

