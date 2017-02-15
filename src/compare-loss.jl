#!/usr/bin/env julia

# Ok, what is the plan here?

using HDF5
using Distributions

include("model.jl")

function sample_variational_model(mu, sigma, idx, numsamples)
    xs = Array(Float64, numsamples)
    s1 = Array(Float64, length(mu))
    s2 = Array(Float64, length(mu))

    unused1 = Array(Float64, length(mu))
    unused2 = Array(Float64, length(mu))
    unused3 = Array(Float64, length(mu))
    unused4 = Array(Float64, length(mu))
    unused5 = Array(Float64, length(mu))

    for i in 1:numsamples
        rand!(MultivariateNormal(mu, sigma), s1)
        simplex!(length(mu), s2, unused1, unused2, unused3, unused4,
                 unused5, s1)
        xs[i] = s2[idx]
    end

    return xs
end


function renorm_isolator_sample(isoquant, isotw, idx)
    numsamples = size(isoquant, 3)
    n = size(isoquant, 1)
    s = Array(Float64, size(isoquant, 3))
    for i in 1:numsamples
        x = reshape(isoquant[1:end, 1, i], (n,))
        x .*= isotw
        x ./= sum(x)
        s[i] = x[idx]
    end

    return s
end


function main()
    isodata = h5open("isolator-output.h5")
    isoquant = isodata["transcript_quantification"]
    isoscale = isodata["sample_scaling"][1,1:end]
    isotw = [parse(Float64, line)
             for line in eachline(open("transcript-weights.txt"))]

    extdata = h5open("sample-data.h5")
    mu = read(extdata["mu"])
    sigma = read(extdata["sigma"])

    lastsample = isoquant[1:end, 1, end]
    @show length(lastsample)

    #ordidx = reverse(sort(1:length(mu), by=i -> mu[i]))
    ordidx = reverse(sort(1:length(lastsample), by=i -> lastsample[i]))
    #idx = ordidx[rand(1:10000)]
    #idx = rand(1:length(ordidx))
    #idx = 16
    idx = 72281
    #@show idx
    #@show isotw[idx]

    #@show mu[16]
    #@show lastsample[16]
    #exit()

    #@show ordidx[1:100]
    #ordidx = reverse(sort(1:length(mu), by=i -> mu[i]))
    #@show ordidx[1:100]

    #idx = rand(1:length(mu))

    numsamples = size(isoquant, 3)
    xs = sample_variational_model(mu, sigma, idx, numsamples)
    ys = renorm_isolator_sample(isoquant, isotw, idx)
    #ys = reshape(isoquant[idx, 1, 1:end], (numsamples,)) ./ isoscale

    @show mean(xs)
    @show mean(ys)

    out = open("loss-data.txt", "w")
    for x in xs
        @printf(out, "extruder,%.6e\n", x)
    end
    for y in ys
        @printf(out, "isolator,%.6e\n", y)
    end
    close(out)
end


main()



# sum p(x) * log(p(x) / q(x))
# = E_p( log(p(x) / q(x)) )
#
# we don't really have a way of computing p(x) though...

#n = size(isoquant, 3)
#for i in 1:numsamples
    #@show i
    #xs = reshape(isoquant[1,1,1:end], (n,))

    ## Ok, now plot this against samples from simplex transformed normal
#end

#@show typeof(isoquant[1,1,1:end])


