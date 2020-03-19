
using ArgParse


function select_approx_method(method_name::String, tree_method::Symbol)
    if method_name == "optimize"
        return OptimizePTTApprox()
    elseif method_name == "logistic_normal"
        return LogisticNormalApprox()
    elseif method_name == "kumaraswamy_ptt"
        return KumaraswamyPTTApprox(tree_method)
    elseif method_name == "logit_skew_normal_ptt"
        return LogitSkewNormalPTTApprox(tree_method)
    elseif method_name == "logit_normal_ptt"
        return LogitNormalPTTApprox(tree_method)
    elseif method_name == "normal_ilr"
        return NormalILRApprox(tree_method)
    elseif method_name == "normal_alr"
        return NormalALRApprox()
    else
        error("$(method_name) is not a know approximation method.")
    end
end

"""
Read each non-empty line from a file into a set. Return an empty set if the
filename is nothing.
"""
function read_line_set(filename)
    line_set = Set{String}()
    if filename != nothing
        open(filename) do input
            for line in eachline(input)
                if !isempty(line)
                    push!(line_set, line)
                end
            end
        end
    end
    return line_set
end


# These are global just so they can get precompiled. ArgParse is slow at
# parsing these arg tables.
const arg_settings = ArgParseSettings()
arg_settings.prog = "polee"
arg_settings.commands_are_required = true

@add_arg_table arg_settings begin
    "prep"
        help = "Approximate likelihood for every sample in an experiment."
        action = :command
    "prep-sample"
        help = "Approximate likelihood for one sample."
        action = :command
    "sample"
        help = "Sample from approximate likelihood."
        action = :command
    "debug-sample"
        help = "Generate samples from a gibbs sampler for diagnostics/benchmarking."
        action = :command
    "debug-optimize"
        help = "Find ML point estimates using expectation maximization for diagnostics/benchmarking."
        action = :command
    "path"
        help = "Print the package path and exit."
        action = :command
    "model"
        help = "Run one models included with polee."
        action = :command
end

@add_arg_table arg_settings["prep"] begin
    "experiment"
        metavar = "experiment.yml"
        help = "Experiment specification file."
        required = true
    "--force"
        help = "Overwrite output files if they already exist."
        action = :store_true
    "--num-threads", "-t" # handled by the wrapper script
        metavar = "N"
        help = "Number of threads to use. (Defaults to number of cores.)"
        required = false
end

@add_arg_table arg_settings["prep-sample"] begin
    "--output", "-o"
        metavar = "prepared-sample.h5"
        help = "Output filename."
        default = "prepared-sample.h5"
    "--exclude-seqs"
        metavar = "excluded-seqs.txt"
        help = "file with list (one per line) of sequence names to exclude."
        required = false
    "--exclude-transcripts"
        metavar = "excluded-transcripts.txt"
        help = "File with list (one per line) of transcript ids to exclude."
        required = false
    "--likelihood-matrix"
        metavar = "likelihood-matrix.h5"
        help = "Output intermediate likelihood matrix. (Used by 'polee gibbs-sample')"
        required = false
    "--approx-method"
        help = "Likelihood approximation method. (Don't mess with this)"
        default = "logit_skew_normal_ptt"
    "--tree-method"
        help = "Tree building heurustic for polya tree transform. (Don't mess with this either)"
        default = "cluster"
    "--no-bias"
        help = "Disable bias correction model."
        action = :store_true
    "--seed"
        metavar = "N"
        help = "RNG seed"
        default = 123456789
        arg_type = Int
    "--clip-read-name-mate"
        help = """
            Remove the training '/1' or '/2' from read names that contain them.
            Without this option, pseudoalignments from kallisto will be treated as single-end.
            """
        action = :store_true
    "--no-efflen-jacobian"
        help = """
            By default the likelihood function includes the jacobian
            determinant of the effective length adjustment, so it is a function
            over transcript expression, rather that length weighted expression.
            This option excludes that jacobian determinant factor.
        """
        action = :store_true
    "--alt-frag-model"
        help = """
            Use a somewhat different fragment model. This alternative model has
            a different set of assumptions, that reads from very short transcripts
            are more common. The end result, on average, is to decrease
            estimated expression of very short transcripts with observed
            reads.
            """
        action = :store_true
    "--verbose"
        help = "Print some additional diagnostic output."
        action = :store_true
    "--dump-bias-training-examples"
        help = "Write training examples collected for use with bias model training."
        action = :store_true
    "--num-threads", "-t" # handled by the wrapper script
        metavar = "N"
        help = "Number of threads to use. (Defaults to number of cores.)"
        required = false
    "--skip-likelihood-approximation"
        help = """Don't approximate likelihood. (Useful in conjuction with
        --likelihood-matrix, if that's all that is needed.)"""
        action = :store_true
    "genome_filename"
        metavar = "sequences.fa"
        help = "Reference sequences in FASTA format."
        required = true
    "reads_filename"
        metavar = "reads.bam"
        help = "Aligned reads in BAM format."
        required = true
    "annotations_filename"
        metavar = "annotations.gff3"
        help = """Transcript annotation filename in GFF3 format, if
        genome alignments are used"""
        required = false
end

@add_arg_table arg_settings["sample"] begin
    "--output", "-o"
        arg_type = String
        default = "post-mean.csv"
    "--kallisto"
        help = """Output samples in a format compatible with kallisto,
        for use with sleuth. """
        action = :store_true
    "--exclude-transcripts"
        metavar = "excluded-transcripts.txt"
        help = "Optional list of transcripts to exclude."
        required = false
    "--num-samples"
        metavar = "N"
        help = "Number of samples to generate."
        default = 1000
        arg_type = Int
    "--annotations"
        metavar = "annotations.gff3"
        help = """Transcript annotations if genome alignments were used.
        If not provided, the command will attempt to use the transcript
        file used during preparation."""
        required = false
    "--sequences"
        metavar = "sequences.fa"
        help = ""
        required = false
    "--trim-prefix"
        metavar = "prefix"
        help = """Trim the given prefix from transcripts identifiers when
        writing output."""
        default = nothing
    "--sample-counts"
        help = """Sample read counts in addition to sampling expression.
        (Default behavior is to sample expression, and compute expected
        read counts if needed."""
        action = :store_true
    "--uniform-gene-prior"
        help = """Sample with a uniform ("noninformative") prior over gene
        expression. (Default behavior is to use a uniform prior over transcript
        expression.)"""
        action = :store_true
    "prepared_sample"
        help = "Prepared RNA-Seq sample to generate samples from."
        metavar = "prepared-sample.h5"
        required = true
end

@add_arg_table arg_settings["debug-sample"] begin
    "--output", "-o"
    "--kallisto"
        help = """Output samples in a format compatible with kallisto,
        for use with sleuth. """
        action = :store_true
    "--num-samples"
        metavar = "N"
        help = "Number of samples to generate and record."
        default = 1000
        arg_type = Int
    "--stride"
        metavar = "N"
        help = "Number of samplet to generate and not record for each
        recorded sample."
        default = 25
        arg_type = Int
    "--burnin"
        metavar = "N"
        help = "Number of initialization samples to generate."
        default = 2000
        arg_type = Int
    "--annotations"
        metavar = "annotations.gff3"
        help = """Load transcripts from annotations file."""
        required = false
    "--sequences"
        metavar = "sequences.fa"
        help = "Load transcripts from sequences file."
        required = false
    "--no-efflen"
        help = "Do not do effective length transformation."
        action = :store_true
    "likelihood-matrix"
        metavar = "likelihood-matrix.h5"
        help = """
            Likelihood matrix as generated with the 'prep' or 'prep-sample'
            commands using the '--likelihood-matrix' argument.
            """
        required = true
end

@add_arg_table arg_settings["debug-optimize"] begin
    "--output", "-o"
        metavar = "estimates.csv"
        default = "estimates.csv"
    "--num-threads", "-t" # handled by the wrapper script
        metavar = "N"
        help = "Number of threads to use. (Defaults to number of cores.)"
        required = false
    "--annotations"
        metavar = "annotations.gff3"
        help = """Load transcripts from annotations file."""
        required = false
    "--sequences"
        metavar = "sequences.fa"
        help = "Load transcripts from sequences file."
        required = false
    "likelihood-matrix"
        metavar = "likelihood-matrix.h5"
        help = """
            Likelihood matrix as generated with the 'prep' or 'prep-sample'
            commands using the '--likelihood-matrix' argument.
            """
        required = true
end


function main(args=nothing)
    if args === nothing
        parsed_args = parse_args(arg_settings)
    else
        parsed_args = parse_args(args, arg_settings)
    end

    command = parsed_args["%COMMAND%"]::String
    command_args = parsed_args[command]::Dict{String, Any}
    if command == "prep"
        polee_prep(command_args)
    elseif command == "prep-sample"
        polee_prep_sample(command_args)
    elseif command == "sample"
        polee_sample(command_args)
    elseif command == "debug-sample"
        polee_debug_sample(command_args)
    elseif command == "debug-optimize"
        polee_debug_optimize(command_args)
    elseif command == "path"
        print(joinpath(dirname(pathof(Polee)), ".."))
    else
        # argument parser should catch this
        error("no command specified")
    end

#=
    elseif subcmd == "estimate" || subcmd == "est"
        @add_arg_table arg_settings begin
            "--output", "-o"
            "--output-format", "-F"
                default = "csv"
            "--exclude-transcripts"
                required = false
            "--informative-prior"
                action = :store_true
            "--credible-lower"
                default = 0.025
                arg_type = Float64
            "--credible-upper"
                default = 0.975
                arg_type = Float64
            "--num-mixture-components"
                default = 12
                arg_type = Int
            "--num-components"
                default = 2
                arg_type = Int
            "--neural-network"
                action = :store_true
            "--batch-size"
                default = 20
                arg_type = Int
            "--output-pca-w"
            "--inference"
                default = "default"
            "--max-num-samples"
                required = false
                arg_type = Int
            "--transcripts"
                required = false
            "--no-gff-hash-check"
                action = :store_true
            "--num-threads", "-t" # handled by the wrapper script
            "--no-gpu"            # handled by the wrapper script
            "feature"
                required = true
            "model"
                required = true
            "experiment"
                required = true
        end

        parsed_args = parse_args(subcmd_args, arg_settings)

        excluded_transcripts = Set{String}()
        if parsed_args["exclude-transcripts"] != nothing
            open(parsed_args["exclude-transcripts"]) do input
                for line in eachline(input)
                    push!(excluded_transcripts, chomp(line))
                end
            end
        end

        spec = YAML.load_file(parsed_args["experiment"])
        if isempty(spec)
            error("Experiment specification is empty.")
        end

        if !haskey(spec, "samples")
            error("Experiment specification has no samples.")
        end

        prep_file_suffix = get(spec, "prep_file_suffix", ".likelihood.h5")

        if "transcripts" ∈ keys(parsed_args) && parsed_args["transcripts"] != nothing
            transcripts_filename = parsed_args["transcripts"]
        elseif haskey(spec, "annotations")
            transcripts_filename = spec["annotations"]
        else
            first_sample = first(spec["samples"])
            if haskey(first_sample, "file")
                first_sample_file = first_sample["file"]
            else
                if !haskey(first_sample, "name")
                    error("Sample in experiment specification is missing a 'name' field.")
                end
                first_sample_file = string(first_sample["name"], prep_file_suffix)
            end

            transcripts_filename =
                read_transcripts_filename_from_prepared(first_sample_file)
            println("Using transcripts file: ", transcripts_filename)
        end

        init_python_modules()

        ts, ts_metadata = Transcripts(transcripts_filename, excluded_transcripts)
        gene_db = write_transcripts("genes.db", ts, ts_metadata)

        if parsed_args["informative-prior"]
            global INFORMATIVE_PRIOR
            INFORMATIVE_PRIOR = true
        end

        model_name = parsed_args["model"]

        batch_size = BATCH_MODEL[model_name] ? parsed_args["batch-size"] : nothing
        max_num_samples =  get(parsed_args, "max-num-samples", nothing)
        loaded_samples = load_samples_from_specification(
            spec, ts, ts_metadata, max_num_samples, batch_size,
            check_gff_hash=!parsed_args["no-gff-hash-check"])

        inference = Symbol(parsed_args["inference"])
        feature = Symbol(parsed_args["feature"])

        output_format = Symbol(parsed_args["output-format"])
        if output_format != :csv && output_format != :sqlite3
            error("Output format must be either \"csv\" or \"sqlite3\".")
        end

        credible_interval =
            (Float64(parsed_args["credible-lower"]),
             Float64(parsed_args["credible-upper"]))

        input = ModelInput(
            loaded_samples, inference, feature, ts, ts_metadata,
            parsed_args["output"], output_format, gene_db,
            credible_interval, parsed_args)

        POLEE_MODELS[model_name](input)

        return
    end
    =#
end


"""
Handle 'polee prep' command.
"""
function polee_prep(parsed_args::Dict{String, Any})
    force_overwrite = parsed_args["force"]
    spec = YAML.load_file(parsed_args["experiment"])

    excluded_transcripts = Set{String}() # TODO: read this
    excluded_seqs = Set{String}() # TODO: read this
    no_bias = Bool(get(spec, "no_bias", false))

    approximation_method_name =
        get(spec, "approximation", "logit_skew_normal_ptt")
    approximation_tree_method =
        Symbol(get(spec, "approximation_tree_method", "cluster"))
    approximation = select_approx_method(
        approximation_method_name, approximation_tree_method)

    reads_file_suffix = get(spec, "reads_file_suffix", nothing)
    prep_file_suffix = get(spec, "prep_file_suffix", ".likelihood.h5")

    if haskey(spec, "reads_decompress_cmd")
        reads_decompress_cmd = split(spec["reads_decompress_cmd"])
    else
        reads_decompress_cmd = nothing
    end

    if haskey(spec, "genome") && haskey(spec, "annotations")
        ts, ts_metadata = Transcripts(spec["annotations"], excluded_transcripts)
        read_transcript_sequences!(ts, spec["genome"])
        sequences_filename = spec["genome"]

    elseif haskey(spec, "transcripts")
        reader = open(FASTA.Reader, transcript_sequence_filename)
        entry = eltype(reader)()
        transcripts = Transcript[]
        while !isnull(tryread!(reader, entry))
            seqname = FASTA.identifier(entry)
            seq = FASTA.sequence(entry)
            id = length(transcripts) + 1
            t = Transcript(seqname, 1, length(seq), STRAND_POS,
                    TranscriptMetadata(seqname, id, [Exon(1, length(seq))], seq))
            push!(transcripts, t)
        end

        ts = Transcripts(transcripts, true)
        ts_metadata = TranscriptsMetadata()
        sequences_filename = spec["transcripts"]
    else
        error(
            """
            Either 'transcripts' or 'genome' and 'annotations' must be specified
            in the experiment specification.
            """)
    end

    sequences_file_hash = SHA.sha1(open(sequences_filename))

    if !haskey(spec, "samples")
        @warn "No samples specified the experiment specification."
        exit(1)
    end

    samples = spec["samples"]
    for sample in samples
        if !haskey(sample, "name")
            error("Sample missing a 'name' field")
        end
        sample_name = sample["name"]
        println(sample_name)
        println(repeat("-", length(sample_name)))

        if haskey(sample, "reads_file")
            reads_file = sample["reads_file"]
        elseif reads_file_suffix !== nothing
            reads_file = string(sample_name, reads_file_suffix)
        else
            error("Sample has no 'reads_file' and there is no 'reads_file_suffix'")
        end

        if haskey(sample, "file")
            output_file = sample["file"]
        else
            output_file = string(sample_name, prep_file_suffix)
        end

        if !force_overwrite && stat(output_file).mtime > stat(reads_file).mtime
            println("Output file is newer than input. Skipping. (Use '--force' to override.")
            continue
        end

        if reads_decompress_cmd !== nothing
            reads_input =
                open(Cmd(Vector{String}(vcat(reads_decompress_cmd, reads_file))))
        else
            reads_input = reads_file
        end

        sample = RNASeqSample(
            ts, ts_metadata, reads_input,
            excluded_seqs, excluded_transcripts,
            sequences_filename, sequences_file_hash,
            no_bias=no_bias)

        approximate_likelihood(approximation, sample, output_file)
    end
end


"""
Handle 'polee prep-sample' command.
"""
function polee_prep_sample(parsed_args::Dict{String, Any})
    if parsed_args["verbose"]
        global_logger(ConsoleLogger(stderr, Logging.Debug))
    end

    tree_method = Symbol(parsed_args["tree-method"])
    approx = select_approx_method(parsed_args["approx-method"], tree_method)

    excluded_seqs = read_line_set(parsed_args["exclude-seqs"])
    excluded_transcripts = read_line_set(parsed_args["exclude-transcripts"])

    Random.seed!(parsed_args["seed"])

    if parsed_args["annotations_filename"] === nothing
        @debug "No transcripts file, assuming transcriptome alignments."
        sample = RNASeqSample(
            parsed_args["genome_filename"],
            parsed_args["reads_filename"],
            excluded_seqs,
            excluded_transcripts,
            parsed_args["likelihood-matrix"] == nothing ?
                Nullable{String}() :
                Nullable(parsed_args["likelihood-matrix"]),
            no_bias=parsed_args["no-bias"],
            dump_bias_training_examples=parsed_args["dump-bias-training-examples"],
            clip_read_name_mate=parsed_args["clip-read-name-mate"],
            alt_frag_model=parsed_args["alt-frag-model"])
    else
        sample = RNASeqSample(
            parsed_args["annotations_filename"],
            parsed_args["genome_filename"],
            parsed_args["reads_filename"],
            excluded_seqs,
            excluded_transcripts,
            parsed_args["likelihood-matrix"] == nothing ?
                Nullable{String}() :
                Nullable(parsed_args["likelihood-matrix"]),
            no_bias=parsed_args["no-bias"],
            dump_bias_training_examples=parsed_args["dump-bias-training-examples"],
            clip_read_name_mate=parsed_args["clip-read-name-mate"],
            alt_frag_model=parsed_args["alt-frag-model"])
    end

    if !parsed_args["skip-likelihood-approximation"]
        approximate_likelihood(
            approx, sample, parsed_args["output"],
            gene_noninformative=false,
            use_efflen_jacobian=!parsed_args["no-efflen-jacobian"])
    end
end


"""
Handle 'polee sample' command.
"""
function polee_sample(parsed_args::Dict{String, Any})
    excluded_transcripts = read_line_set(
        parsed_args["exclude-transcripts"])

    input = h5open(parsed_args["prepared_sample"])
    input_metadata = g_open(input, "metadata")
    check_prepared_sample_version(input_metadata)
    transcripts_filename = read(attrs(input_metadata)["gfffilename"])
    sequences_filename = read(attrs(input_metadata)["fafilename"])
    close(input_metadata)

    if parsed_args["annotations"] !== nothing
        transcripts_filename = parsed_args["annotations"]
    end

    if isempty(transcripts_filename)
        if parsed_args["sequences"] !== nothing
            sequences_filename = parsed_args["sequences"]
        end

        if isempty(sequences_filename)
            error("""Either '--sequences' (if transcriptome aligments were used) or
            '--transcripts' (if genome alignments were used) must be
            given.""")
        end

        ts, ts_metadata = read_transcripts_from_fasta(
            sequences_filename, excluded_transcripts)
    else
        ts, ts_metadata = Transcripts(
            transcripts_filename, excluded_transcripts)
    end

    tnames = String[t.metadata.name for t in ts]
    if parsed_args["trim-prefix"] !== nothing
        for (i, tname) in enumerate(tnames)
            tnames[i] = replace(
                tname, parsed_args["trim-prefix"] => "")
        end
    end

    n = length(ts)

    node_parent_idxs = read(input["node_parent_idxs"])
    node_js          = read(input["node_js"])
    efflens          = read(input["effective_lengths"])
    m                = read(input["m"])

    mu    = read(input["mu"])
    sigma = exp.(read(input["omega"]))
    alpha = read(input["alpha"])

    t = PolyaTreeTransform(node_parent_idxs, node_js)

    num_samples = parsed_args["num-samples"]
    samples = Array{Float32}(undef, (num_samples, n))

    als = ApproxLikelihoodSampler()
    set_transform!(als, t, mu, sigma, alpha)
    xs = Array{Float32}(undef, n)

    prog = Progress(num_samples, 0.25, "Sampling from approx. likelihood ", 60)
    for i in 1:num_samples
        next!(prog)
        rand!(als, xs)

        # effective length transform
        xs ./= efflens
        xs ./= sum(xs)
        samples[i, :] = xs
    end
    finish!(prog)

    post_mean = mean(samples, dims=1)[1,:]

    function expected_counts(prop)
        prop_ = prop .* efflens
        prop_ ./= sum(prop_)
        return Vector{Float64}(prop_ .* m)
    end

    function sample_counts(prop_cumsum, num_reads)
        counts = zeros(Float64, n)
        r = rand()
        for _ in 1:num_reads
            r = rand()
            i = min(n, searchsortedfirst(prop_cumsum, r))
            counts[i] += 1
        end
        return counts
    end

    if parsed_args["sample-counts"]
        prop_to_counts = prop -> sample_counts(cumsum((prop.*efflens) ./= sum(prop.*efflens)), m)
    else
        prop_to_counts = expected_counts
    end

    if parsed_args["kallisto"]
        filename = parsed_args["output"] === nothing ?
            "polee-sample.h5" : parsed_args["output"]

        h5open(filename, "w") do output
            output["est_counts"] = prop_to_counts(post_mean)

            aux_group = g_create(output, "aux")
            aux_group["num_bootstrap"]    = Int[num_samples]
            aux_group["eff_lengths"]      = Vector{Float64}(efflens)
            aux_group["lengths"]          = Int[exonic_length(t) for t in ts]
            aux_group["ids"]              = tnames
            aux_group["call"]             = String[join(ARGS, " ")]
            aux_group["index_version"]    = Int[-1]
            aux_group["kallisto_version"] = "polee sample" # TODO: should record polee version
            aux_group["start_time"]       = string(now())

            bootstrap_group = g_create(output, "bootstrap")
            for i in 1:num_samples
                bootstrap_group[string("bs", i-1)] = prop_to_counts(samples[i, :])
            end
        end
    else
        filename = parsed_args["output"] === nothing ?
            "polee-sample.csv" : parsed_args["output"]

        open(parsed_args["output"], "w") do output
            println(output, "transcript_id,tpm")
            for (j, t) in enumerate(ts)
                println(output, tnames[j], ",", 1e6*post_mean[j])
            end
        end
    end
end


"""
Handle 'polee debug-sample' command.
"""
function polee_debug_sample(parsed_args::Dict{String, Any})

    if (parsed_args["annotations"] !== nothing && parsed_args["sequences"] !== nothing) ||
        (parsed_args["annotations"] === nothing && parsed_args["sequences"] === nothing)
        error("Exactly one of --annotations and --sequences must be given.")
    end

    if parsed_args["annotations"] !== nothing
        ts, ts_metadata = Transcripts(parsed_args["annotations"])
    end

    if parsed_args["sequences"] !== nothing
        ts, ts_metadata = read_transcripts_from_fasta(
            parsed_args["sequences"], Set{String}())
    end

    output_filename =
        parsed_args["output"] !== nothing ? parsed_args["output"] :
        parsed_args["kallisto"] ? "gibbs-samples.h5" : "gibbs-samples.csv"

    gibbs_sampler(
        parsed_args["likelihood-matrix"], output_filename, ts,
        kallisto=parsed_args["kallisto"],
        num_samples=parsed_args["num-samples"],
        num_burnin_samples=parsed_args["burnin"],
        sample_stride=parsed_args["stride"],
        use_efflen=!parsed_args["no-efflen"])
end


"""
Handle the 'debug-optimize' command.
"""
function polee_debug_optimize(parsed_args::Dict{String, Any})
    if (parsed_args["annotations"] !== nothing && parsed_args["sequences"] !== nothing) ||
        (parsed_args["annotations"] === nothing && parsed_args["sequences"] === nothing)
        error("Exactly one of --annotations and --sequences must be given.")
    end

    if parsed_args["annotations"] !== nothing
        ts, ts_metadata = Transcripts(parsed_args["annotations"])
    end

    if parsed_args["sequences"] !== nothing
        ts, ts_metadata = read_transcripts_from_fasta(
            parsed_args["sequences"], Set{String}())
    end

    transcript_names = String[t.metadata.name for t in ts]

    tpms = expectation_maximization(parsed_args["likelihood-matrix"])

    open(parsed_args["output"], "w") do output
        println(output, "transcript_id,tpm")
        for j in 1:length(tpms)
            println(output, transcript_names[j], ",", tpms[j])
        end
    end
end

precompile(main, ())

