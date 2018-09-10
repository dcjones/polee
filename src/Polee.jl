#!/usr/bin/env julia

module Polee

__precompile__(false)

using PyCall
using Pkg

@pyimport tensorflow as tf
@pyimport tensorflow.contrib.distributions as tfdist
@pyimport tensorflow.contrib.tfprof as tfprof
@pyimport tensorflow.python.client.timeline as tftl
@pyimport tensorflow.python.util.all_util as tfutil
@pyimport edward as ed
@pyimport edward.models as edmodels
@pyimport edward.util as edutil
tfutil.reveal_undocumented("edward.util.graphs")
@pyimport edward.util.graphs as edgraphs

# check if the tensorflow extension exists and bulid it if it doesn't.
function build_tensorflow_ext_if_needed()
    ext_path = joinpath(dirname(pathof(Polee)), "tensorflow_ext")
    ext_lib_filename = joinpath(ext_path, "hsb_ops.so")
    ext_src_filename = joinpath(ext_path, "hsb_ops.cpp")
    if !isfile(ext_lib_filename)
        @warn "Attempting to build tensorflow extension automatically."
        cflags = chomp(String(read(`python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'`)))
        lflags = chomp(String(read(`python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'`)))
        cmd = "g++ -std=c++11 -shared -g -O2 $(ext_src_filename) -o $(ext_lib_filename) -fPIC $(cflags) $(lflags)"
        println(cmd)
        run(Cmd(Vector{String}(split(cmd))))
    end
end

build_tensorflow_ext_if_needed()
pushfirst!(PyVector(pyimport("sys")["path"]), dirname(pathof(Polee)))
@pyimport polee as polee_py

# TODO: Once PyCall supports overloading dot, we should use this version
# and enable precompilation.
# using PyCall

# const tf       = PyNULL()
# const tfdist   = PyNULL()
# const tfprof   = PyNULL()
# const tftl     = PyNULL()
# const tfutil   = PyNULL()
# const ed       = PyNULL()
# const edmodels = PyNULL()
# const edutil   = PyNULL()
# const polee_py = PyNULL()

# function __init__()
#     println("Polee.__init__")

#     copy!(tf,       pyimport("tensorflow"))
#     copy!(tfdist,   pyimport("tensorflow.contrib.distributions"))
#     copy!(tfprof,   pyimport("tensorflow.contrib.tfprof"))
#     copy!(tftl,     pyimport("tensorflow.python.client.timeline"))
#     copy!(tfutil,   pyimport("tensorflow.python.util.all_util"))
#     copy!(ed,       pyimport("edward"))
#     copy!(edmodels, pyimport("edward.models"))
#     copy!(edutil,   pyimport("edward.util"))
#     tfutil.reveal_undocumented("edward.util.graphs")

#     pushfirst!(PyVector(pyimport("sys")["path"]), dirname(pathof(Polee)))
#     copy!(polee_py, pyimport("polee"))

#     build_tensorflow_ext_if_needed()
# end



using ArgParse
using GenomicFeatures
using BioAlignments
using BioSequences
using DataStructures
using Distributions
using HDF5
using ProgressMeter
using SQLite
using StatsBase
using Nullables
import IntervalTrees
import SHA
import YAML

using Base64: base64encode
using SparseArrays: findnz, sparse, SparseMatrixCSC
using Printf: @printf, @sprintf
using Random
# using Profile
# using InteractiveUtils


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


include("constants.jl")
include("sparse.jl")
include("reads.jl")
include("transcripts.jl")
include("bias.jl")
include("fragmodel.jl")
include("model.jl")
include("sample.jl")
include("kumaraswamy.jl")
include("logitnormal.jl")
include("sinh_arcsinh.jl")
include("likelihood-approximation.jl")
include("estimate.jl")
include("gibbs.jl")
include("em.jl")
include("stick_breaking.jl")
include("isometric_log_ratios.jl")
include("additive_log_ratios.jl")
include("sequences.jl")
include("evaluate.jl")

# TODO: automate including everything under models
POLEE_MODELS = Dict{String, Function}()
include("models/splicing.jl")
include("models/linear-regression.jl")
include("models/simple-linear-regression.jl")
include("models/simple-mode.jl")
include("models/simple-pca.jl")
include("models/logistic-regression.jl")
include("models/simple-logistic-regression.jl")
include("models/quantification.jl")
include("models/pca.jl")
include("models/gplvm.jl")

include("main.jl")

end # module Polee
