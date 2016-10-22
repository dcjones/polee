
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


