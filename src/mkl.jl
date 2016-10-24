
# Sparse matrix multiplication with MKL


const libmkl_path = "/opt/intel/compilers_and_libraries_2017.0.098/linux/mkl/lib/intel64_lin/libmkl_rt.so"


function mkl_A_mul_B!(ans::Vector{Float32}, A::SparseMatrixCSC{Float32},
                      b::Vector{Float32})
    ccall((:mkl_scscmv, libmkl_path), Void,
          (Cstring,       # transa
           Ptr{Int},      # m
           Ptr{Int},      # k
           Ptr{Float32},  # alpha
           Cstring,       # matdescra
           Ptr{Float32},  # val
           Ptr{Int},      # indx
           Ptr{Int},      # pntrb
           Ptr{Int},      # pntre
           Ptr{Float32},  # x
           Ptr{Float32},  # beta
           Ptr{Float32}), # y
          "N", Ref(A.m), Ref(A.n), Ref(1.0f0), "GXXF", A.nzval, A.rowval, A.colptr,
          pointer(A.colptr, 2), b, Ref(0.0f0), ans)
end


function mkl_At_mul_B!(ans::Vector{Float32}, A::SparseMatrixCSC{Float32},
                       b::Vector{Float32})
    ccall((:mkl_scscmv, libmkl_path), Void,
          (Cstring,       # transa
           Ptr{Int},      # m
           Ptr{Int},      # k
           Ptr{Float32},  # alpha
           Cstring,       # matdescra
           Ptr{Float32},  # val
           Ptr{Int},      # indx
           Ptr{Int},      # pntrb
           Ptr{Int},      # pntre
           Ptr{Float32},  # x
           Ptr{Float32},  # beta
           Ptr{Float32}), # y
          "T", Ref(A.m), Ref(A.n), Ref(1.0f0), "GXXF", A.nzval, A.rowval, A.colptr,
          pointer(A.colptr, 2), b, Ref(0.0f0), ans)
end



# Using the inpect-execute model

immutable MKLSparseMatrixCSC
    A::SparseMatrixCSC
    ptr::Ref{Ptr{Void}} # sparse_matrix_t
end


function MKLSparseMatrixCSC(A::SparseMatrixCSC)
    handle = Ref{Ptr{Void}}(C_NULL)
    status = ccall((:mkl_sparse_s_create_csc, libmkl_path),
                   Cint, (Ptr{Ptr{Void}}, # handle
                          Cint,           # indexing
                          Int,            # rows
                          Int,            # cols
                          Ptr{Int},       # pntrb
                          Ptr{Int},       # pntre
                          Ptr{Int},       # index
                          Ptr{Float32}),  # val
                   handle, 1, A.m, A.n, A.colptr, pointer(A.colptr, 2),
                   A.rowval, A.nzval)
    if status != 0
        error("MKL sparse matrix init failed with code $(status)")
    end

    #status = ccall((:mkl_sparse_set_memory_hint, libmkl_path),
                   #Cint, (Ptr{Void}, Cint),
                   #handle.x, 81)

    #status = ccall((:mkl_sparse_set_mv_hint, libmkl_path),
                   #Cint, (Ptr{Void},
                          #Cint, # operation
                          #MKLMatrixDescr,
                          #Int), # expected calls
                   #handle.x, SPARSE_OPERATION_NON_TRANSPOSE,
                   #MKLMatrixDescr(20, 0, 0), 50)

    #if status != 0
        #error("MKL sparse hint failed with code $(status)")
    #end

    #status = ccall((:mkl_sparse_optimize, libmkl_path),
                   #Cint, (Ptr{Void},), handle.x)
    #if status != 0
        #error("MKL sparse matrix optimization failed with code $(status)")
    #end

    return MKLSparseMatrixCSC(A, handle)
end


immutable MKLMatrixDescr
    typ::Cint
    mode::Cint
    diag::Cint
end

const SPARSE_OPERATION_NON_TRANSPOSE       = 10
const SPARSE_OPERATION_TRANSPOSE           = 11
const SPARSE_OPERATION_CONJUGATE_TRANSPOSE = 12

const SPARSE_MATRIX_TYPE_GENERAL            = 20
const SPARSE_MATRIX_TYPE_SYMMETRIC          = 21
const SPARSE_MATRIX_TYPE_HERMITIAN          = 22
const SPARSE_MATRIX_TYPE_TRIANGULAR         = 23
const SPARSE_MATRIX_TYPE_DIAGONAL           = 24
const SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR   = 25
const SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL     = 26


function Base.A_mul_B!(ans::Vector{Float32}, A::MKLSparseMatrixCSC,
                       b::Vector{Float32})
    status = ccall((:mkl_sparse_s_mv, libmkl_path),
                   Cint, (Cint,          # transA
                          Float32,       # alpha
                          Ptr{Void},     # A
                          MKLMatrixDescr,
                          Ptr{Float32},  # x
                          Float32,       # beta
                          Ptr{Float32}), # y
                   SPARSE_OPERATION_NON_TRANSPOSE,
                   1.0, A.ptr.x,
                   MKLMatrixDescr(20, 0, 0),
                   b, 0.0, ans)
    if status != 0
        error("MKL sparse matrix multiplication failed with code $(status)")
    end
    return ans
end


function Base.At_mul_B!(ans::Vector{Float32}, A::MKLSparseMatrixCSC,
                       b::Vector{Float32})
    status = ccall((:mkl_sparse_s_mv, libmkl_path),
                   Cint, (Cint,          # transA
                          Float32,       # alpha
                          Ptr{Void},     # A
                          MKLMatrixDescr,
                          Ptr{Float32},  # x
                          Float32,       # beta
                          Ptr{Float32}), # y
                   SPARSE_OPERATION_TRANSPOSE,
                   1.0, A.ptr.x,
                   MKLMatrixDescr(20, 0, 0),
                   b, 0.0, ans)
    if status != 0
        error("MKL sparse matrix multiplication failed with code $(status)")
    end
    return ans
end

