

#const mkllib = "/opt/intel/compilers_and_libraries_2019.2.187/linux/mkl/lib/intel64_lin/libmkl_rt.so"
const mkllib = "/opt/intel/mkl/lib/intel64/libmkl_rt.so"

const LAPACK_ROW_MAJOR = 101
const LAPACK_COL_MAJOR = 102

function mkl_cholesky!(X::Matrix{Float32})
    n = size(X, 1)
    @assert size(X, 1) == size(X, 2)
    info = ccall(
        (:LAPACKE_spotrf, mkllib), Cint,
        (Cint, Cchar, Cint, Ptr{Float32}, Cint),
        LAPACK_COL_MAJOR, 'U', n, X, n)
    return Cholesky{Float32, Matrix{Float32}}(X, 'U', info)
end

function mkl_cholesky!(X::Matrix{Float64})
    n = size(X, 1)
    @assert size(X, 1) == size(X, 2)
    info = ccall(
        (:LAPACKE_dpotrf, mkllib), Cint,
        (Cint, Cchar, Cint, Ptr{Float64}, Cint),
        LAPACK_COL_MAJOR, 'U', n, X, n)
    return Cholesky{Float64, Matrix{Float64}}(X, 'U', info)
end
