
using HDF5

include("fastmath.jl")
include("mkl.jl")
include("model.jl")


function main()
    input = h5open("output.h5", "r")
    m = read(input["m"]) # number of reads
    n = read(input["n"]) # number of transcripts
    colptr = read(input["colptr"])
    rowval = read(input["rowval"])
    nzval = read(input["nzval"])
    @show (m, n)

    X = SparseMatrixCSC(m, n, colptr, rowval, nzval)
    π = zeros(Float32, n)
    ϕ = zeros(Float32, n)
    grad = zeros(Float32, n)

    model = Model(m, n)


end


main()
