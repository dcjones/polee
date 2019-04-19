#!/usr/bin/env julia

module Polee


using GenomicFeatures
using BioAlignments
using BioSequences
using DataStructures
using Distributions
using HDF5
using ProgressMeter
using StatsBase
using Nullables
using Logging
import IntervalTrees
import SHA
import YAML

using Base64: base64encode, base64decode
using SparseArrays: findnz, sparse, SparseMatrixCSC
using Dates: now
using Printf: @printf, @sprintf
using Statistics
using Random
# using Profile
# using InteractiveUtils

"""
More convenient interface to Bio.jl read! functions.
"""
function tryread!(reader, entry)
    try
        read!(reader, entry)
        return true
    catch err
        if isa(err, EOFError)
            return false
        end
        rethrow()
    end
end

# convenient way to time things
macro tic()
    t0 = esc(:t0)
    quote
        $(t0) = time()
    end
end

macro toc(context)
    t0 = esc(:t0)
    quote
        dt = time() - $(t0)
        @debug @sprintf("%s: %0.2f secs", $(context), dt)
    end
end


include("constants.jl")
include("sparse.jl")
include("reads.jl")
include("transcripts.jl")
include("bias.jl")
include("fragmodel.jl")
include("likelihood.jl")
include("rnaseq_sample.jl")
include("kumaraswamy.jl")
include("logitnormal.jl")
include("sinh_arcsinh.jl")
include("hclust.jl")
include("ptt.jl")
include("likelihood-approximation.jl")
include("gibbs.jl")
include("em.jl")
include("isometric_log_ratios.jl")
include("additive_log_ratios.jl")
include("sequences.jl")
include("evaluate.jl")
include("approx-sampler.jl")



# TODO: need to rethink this wit the move to edward2
# POLEE_MODELS = Dict{String, Function}()
# include("models/linear-regression.jl")
# include("models/simple-linear-regression.jl")
# include("models/simple-mode.jl")
# include("models/simple-pca.jl")
# include("models/logistic-regression.jl")
# include("models/simple-logistic-regression.jl")
# include("models/quantification.jl")
# include("models/pca.jl")
# include("models/gplvm.jl")

include("main.jl")

end # module Polee
