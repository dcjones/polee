#!/usr/bin/env julia

module Polee

using GenomicFeatures
using BioAlignments
using BioSequences
using FASTX
using DataStructures
using Distributions
using HDF5
using ProgressMeter
using StatsBase
using Nullables
using Logging
using SpecialFunctions: beta, digamma # used by kumaraswamy
import XAM: BAM, SAM
import GFF3
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


"""
Install the polee command line script to the given path. If not specified, it will
copy the script to \$HOME/bin.

By default it will not overwrite an existing file. Pass 'force=true' to overwrite.
"""
function install(dest_path=joinpath(ENV["HOME"], "bin"); force=false)
    src = joinpath(dirname(pathof(Polee)), "..", "polee")
    dest = joinpath(dest_path, "polee")
    cp(src, dest, force=force)
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
include("likelihood-approximation-alt.jl")
include("gibbs.jl")
include("em.jl")
include("isometric_log_ratios.jl")
include("additive_log_ratios.jl")
include("sequences.jl")
include("evaluate.jl")
include("approx-sampler.jl")

include("main.jl")

include("PoleeModel.jl")
include("regression.jl")

end # module Polee
