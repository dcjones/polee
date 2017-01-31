

using HDF5
using NLopt

include("fastmath.jl")
using .FastMath

include("mkl.jl")
include("model.jl")


function main()
    #input = h5open("output.h5", "r")
    #m = read(input["m"]) # number of reads
    #n = read(input["n"]) # number of transcripts
    #colptr = read(input["colptr"])
    #rowval = read(input["rowval"])
    #nzval = read(input["nzval"])

    input = open("isolator-weight-matrix.txt")
    n = parse(Int, readline(input))
    m = parse(Int, readline(input))
    @show (m, n)
    I = Array(Int, 0)
    J = Array(Int, 0)
    V = Array(Float32, 0)
    for line in eachline(input)
        j, i, v = split(line, ',')
        push!(I, 1 + parse(Int, i))
        push!(J, 1 + parse(Int, j))
        push!(V, parse(Float32, v))
    end
    X = sparse(I, J, V, m, n)
    #mklX = MKLSparseMatrixCSC(X)


    @show (m, n)

    #X = SparseMatrixCSC(m, n, colptr, rowval, nzval)
    #mklX = MKLSparseMatrixCSC(X)

    π = zeros(Float32, n)
    grad = zeros(Float32, n)

    model = Model(m, n)

    #log_post(model, X, π, grad)
    #@time log_post(model, X, π, grad)
    #exit()

    # optimize!
    opt = Opt(:LD_CCSAQ, n)
    ftol_abs!(opt, 1000)
    initial_step!(opt, 1e-9)
    max_objective!(opt, (π, grad) -> log_likelihood(model, X, π, grad))
    @time optimize(opt, π)

    simplex!(n, model.π_simplex, grad, model.xs_sum, model.zs,
             model.zs_log_sum, π)

    ## check gradient
    #ε = 1e-4
    #grad_ = Array(Float32, n)
    #for j in 1:n
        #πj = π[j]
        #π[j] += ε
        #lp = log_post(model, X, π, grad_)
        #numgrad = (Float64(lp) - Float64(lp0)) / ε
        #π[j] = πj
        #@printf("%d\t%0.4e\t%0.4e\t%0.4f\t%0.4f\n", j, grad[j], numgrad,
                #(grad[j] - numgrad), (grad[j] - numgrad) / (grad[j] + numgrad))
    #end
end


main()
