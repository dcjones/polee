
# Useful function for models based on approximate likelihood.

module PoleeModel

export
    init_python_modules,
    load_samples_from_specification,
    load_transcripts_from_args

import Polee
using Polee: HSBTransform, Transcripts, TranscriptsMetadata, LoadedSamples

import YAML
using PyCall
using SQLite
using HDF5
using Printf: @printf, @sprintf
using Base64: base64encode, base64decode
using ProgressMeter
using Random: shuffle

const polee_py = PyNULL()
const tf = PyNULL()

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

function init_python_modules()
    build_tensorflow_ext_if_needed()
    python_path = PyVector(getproperty(pyimport("sys"), "path"))
    pushfirst!(python_path, joinpath(dirname(pathof(Polee))))
    pushfirst!(python_path, joinpath(dirname(pathof(Polee)), "..", "models"))
    copy!(polee_py, pyimport("polee"))
    copy!(tf, pyimport("tensorflow"))
end


"""
If transcripts are passed specifically as an argument, load them, otherwise
try to determine their location from metadata.
"""
function load_transcripts_from_args(parsed_args, excluded_transcripts=Set{String}())
    spec = YAML.load_file(parsed_args["experiment"])
    prep_file_suffix = get(spec, "prep_file_suffix", ".likelihood.h5")

    transcripts_filename = nothing
    sequences_filename = nothing

    if haskey(spec, "samples") && !isempty(spec["samples"])
        first_sample = spec["samples"][1]
        sample_filename = nothing
        if haskey(first_sample, "file")
            sample_filename = first_sample["file"]
        elseif haskey(first_sample, "name")
            sample_filename = string(first_sample["name"], prep_file_suffix)
        else
            error("Sample in specification has no filename.")
        end

        input = h5open(sample_filename)
        input_metadata = g_open(input, "metadata")
        Polee.check_prepared_sample_version(input_metadata)

        transcripts_filename = read(attrs(input_metadata)["gfffilename"])
        sequences_filename = read(attrs(input_metadata)["fafilename"])

        close(input_metadata)
        close(input)
    end

    if haskey(parsed_args, "annotations") && parsed_args["annotations"] !== nothing
        transcripts_filename = parsed_args["annotations"]
    end

    if transcripts_filename === nothing
        if haskey(parsed_args, "sequences") && parsed_args["sequences"] !== nothing
            sequences_filename = parsed_args["sequences"]
        end

        if sequences_filename === nothing
            error("""Either '--sequences' (if transcriptome aligments were used) or
            '--transcripts' (if genome alignments were used) must be
            given.""")
        end

        return Polee.read_transcripts_from_fasta(
            sequences_filename, excluded_transcripts)
    else
        return Transcripts(
            transcripts_filename, excluded_transcripts)
    end
end


include("estimate.jl")
include("splicing.jl")
include("models.jl")

end

