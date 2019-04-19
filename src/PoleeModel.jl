
# Useful function for models based on approximate likelihood.

module PoleeModel

using PyCall
using SQLite

const polee_py = PyNULL()
const tf = PyNULL()

# @pyimport tensorflow as tf
# @pyimport tensorflow.contrib.distributions as tfdist
# @pyimport tensorflow.contrib.tfprof as tfprof
# @pyimport tensorflow.python.client.timeline as tftl
# @pyimport tensorflow.python.util.all_util as tfutil
# @pyimport edward as ed
# @pyimport edward.models as edmodels
# @pyimport edward.util as edutil
# tfutil.reveal_undocumented("edward.util.graphs")
# @pyimport edward.util.graphs as edgraphs

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

# build_tensorflow_ext_if_needed()
# pushfirst!(PyVector(pyimport("sys")["path"]), dirname(pathof(Polee)))
# @pyimport polee as polee_py

function init_python_modules()
    build_tensorflow_ext_if_needed()
    pushfirst!(PyVector(pyimport("sys")["path"]), dirname(pathof(Polee)))
    copy!(polee_py, pyimport("polee"))
    copy!(tf, pyimport("tensorflow"))
end

include("estimate.jl")
include("splicing.jl")
include("models.jl")

end

