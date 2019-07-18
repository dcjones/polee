
# Useful function for models based on approximate likelihood.

module PoleeModel

export
    init_python_modules,
    load_samples_from_specification,
    load_point_estimates_from_specification,
    load_transcripts_from_args,
    approximate_splicing_likelihood,
    write_transcripts,
    create_tensorflow_variables!

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
using GenomicFeatures
using Statistics

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


"""
Serialize a GFF3 file into sqlite3 database.
"""
function write_transcripts(output_filename, transcripts, metadata)
    db = SQLite.DB(output_filename)

    # Gene Table
    # ----------

    gene_nums = Polee.assign_gene_nums(transcripts, metadata)

    SQLite.execute!(db, "drop table if exists genes")
    SQLite.execute!(db,
        """
        create table genes
        (
            gene_num INT PRIMARY KEY,
            gene_id TEXT,
            gene_name TEXT,
            gene_biotype TEXT,
            gene_description TEXT
        )
        """)

    ins_stmt = SQLite.Stmt(db, "insert into genes values (?1, ?2, ?3, ?4, ?5)")
    SQLite.execute!(db, "begin transaction")
    for (gene_id, gene_num) in gene_nums
        SQLite.bind!(ins_stmt, 1, gene_num)
        SQLite.bind!(ins_stmt, 2, gene_id)
        SQLite.bind!(ins_stmt, 3, get(metadata.gene_name, gene_id, ""))
        SQLite.bind!(ins_stmt, 4, get(metadata.gene_biotype, gene_id, ""))
        SQLite.bind!(ins_stmt, 5, get(metadata.gene_description, gene_id, ""))
        SQLite.execute!(ins_stmt)
    end
    SQLite.execute!(db, "end transaction")

    # Transcript Table
    # ----------------

    SQLite.execute!(db, "drop table if exists transcripts")
    SQLite.execute!(db,
        """
        create table transcripts
        (
            transcript_num INT PRIMARY KEY,
            transcript_id TEXT,
            kind TEXT,
            seqname TEXT,
            strand INT,
            gene_num INT,
            biotype TEXT,
            exonic_length INT
        )
        """)
    ins_stmt = SQLite.Stmt(db,
        "insert into transcripts values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)")
    SQLite.execute!(db, "begin transaction")
    for t in transcripts
        SQLite.bind!(ins_stmt, 1, t.metadata.id)
        SQLite.bind!(ins_stmt, 2, String(t.metadata.name))
        SQLite.bind!(ins_stmt, 3, get(metadata.transcript_kind, t.metadata.name, ""))
        SQLite.bind!(ins_stmt, 4, String(t.seqname))
        SQLite.bind!(ins_stmt, 5,
            t.strand == STRAND_POS ? 1 :
            t.strand == STRAND_NEG ? -1 : 0)
        SQLite.bind!(ins_stmt, 6, gene_nums[metadata.gene_id[t.metadata.name]])
        SQLite.bind!(ins_stmt, 7, get(metadata.transcript_biotype, t.metadata.name, ""))
        SQLite.bind!(ins_stmt, 8, Polee.exonic_length(t))
        SQLite.execute!(ins_stmt)
    end
    SQLite.execute!(db, "end transaction")


    # Exon Table
    # ----------

    SQLite.execute!(db, "drop table if exists exons")
    SQLite.execute!(db,
        """
        create table exons
        (
            transcript_num INT,
            first INT,
            last INT
        )
        """)

    ins_stmt = SQLite.Stmt(db, "insert into exons values (?1, ?2, ?3)")
    SQLite.execute!(db, "begin transaction")
    for t in transcripts
        for exon in t.metadata.exons
            SQLite.bind!(ins_stmt, 1, t.metadata.id)
            SQLite.bind!(ins_stmt, 2, exon.first)
            SQLite.bind!(ins_stmt, 3, exon.last)
            SQLite.execute!(ins_stmt)
        end
    end
    SQLite.execute!(db, "end transaction")

    return db
end


include("estimate.jl")
include("splicing.jl")
include("models.jl")

end
