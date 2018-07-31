#!/usr/bin/env julia

module Polee

using PyCall

unshift!(PyVector(pyimport("sys")["path"]), Pkg.dir("Polee", "src"))
@pyimport tensorflow as tf
@pyimport tensorflow.contrib.distributions as tfdist
@pyimport tensorflow.contrib.tfprof as tfprof
@pyimport tensorflow.python.client.timeline as tftl
@pyimport tensorflow.python.util.all_util as tfutil
@pyimport edward as ed
@pyimport edward.models as edmodels
@pyimport edward.util as edutil
tfutil.reveal_undocumented("edward.util.graphs")
# @pyimport edward.util.graphs as edgraphs

# check if the tensorflow extension exists and bulid it if it doesn't.
function build_tensorflow_ext_if_needed()
    ext_path = Pkg.dir("Polee", "src", "tensorflow_ext")
    ext_lib_filename = joinpath(ext_path, "hsb_ops.so")
    ext_src_filename = joinpath(ext_path, "hsb_ops.cpp")
    if !isfile(ext_lib_filename)
        warn("Attempting to build tensorflow extension automatically.")
        cflags = chomp(String(read(`python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'`)))
        lflags = chomp(String(read(`python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'`)))
        cmd = "g++ -std=c++11 -shared -g -O2 $(ext_src_filename) -o $(ext_lib_filename) -fPIC $(cflags) $(lflags)"
        println(cmd)
        run(Cmd(Vector{String}(split(cmd))))
    end
end

build_tensorflow_ext_if_needed()

@pyimport polee as polee_py

using ArgParse
using GenomicFeatures
using BioAlignments
using BioSequences
using DataStructures
using Distributions
using HDF5
using Interpolations
using ProgressMeter
using SQLite
using StatsBase
using Nullables
import IntervalTrees
import SHA
import YAML

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
