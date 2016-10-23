
using HDF5
using NLopt
using Distributions

include("fastmath.jl")
include("mkl.jl")
include("model.jl")

# TODO: understand what the jacobian term is to handl hypercube -> simple
# transformation


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
    #π = rand(Normal(0, 1), n)
    ϕ = zeros(Float32, n)
    grad = zeros(Float32, n)

    model = Model(m, n)

    #log_post(model, X, π, grad)
    #@time log_post(model, X, π, grad)
    #exit()

    # optimize!
    opt = Opt(:LD_CCSAQ, n)
    ftol_abs!(opt, 1000)
    initial_step!(opt, 1e-9)
    max_objective!(opt, (π, grad) -> log_post(model, X, π, grad))
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
