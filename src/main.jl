

function print_usage()
    println("Usage: polee <command>\n")
    println("where command is one of:")
    println("  likelihood-matrix")
    println("  likelihood-approx")
    println("  prepare-sample")
    println("  prepare-experiment")
    println("  estimate")
end


function select_approx_method(method_name::String, tree_method::Symbol)
    if method_name == "optimize"
        return OptimizeHSBApprox()
    elseif method_name == "logistic_normal"
        return LogisticNormalApprox()
    elseif method_name == "kumaraswamy_hsb"
        return KumaraswamyHSBApprox(tree_method)
    elseif method_name == "logit_skew_normal_hsb"
        return LogitSkewNormalHSBApprox(tree_method)
    elseif method_name == "logit_normal_hsb"
        return LogitNormalHSBApprox(tree_method)
    elseif method_name == "normal_ilr"
        return NormalILRApprox(tree_method)
    elseif method_name == "normal_alr"
        return NormalALRApprox()
    else
        error("$(method_name) is not a know approximation method.")
    end
end


function main()
    Random.seed!(12345678)

    if isempty(ARGS)
        print_usage()
        exit(1)
    end

    subcmd = ARGS[1]
    subcmd_args = ARGS[2:end]
    arg_settings = ArgParseSettings()

    if subcmd == "likelihood-matrix"
        @add_arg_table arg_settings begin
            "--output", "-o"
                default = "likelihood-matrix.h5"
            "transcripts_filename"
                required = true
            "genome_filename"
                required = true
            "reads_filename"
                required = true
            "--excluded-seqs"
                required = false
            "--exclude-transcripts"
                required = false
        end
        parsed_args = parse_args(subcmd_args, arg_settings)

        excluded_seqs = Set{String}()
        if parsed_args["excluded-seqs"] != nothing
            open(parsed_args["excluded-seqs"]) do input
                for line in eachline(input)
                    push!(excluded_seqs, chop(line))
                end
            end
        end

        excluded_transcripts = Set{String}()
        if parsed_args["exclude-transcripts"] != nothing
            open(parsed_args["exclude-transcripts"]) do input
                for line in eachline(input)
                    push!(excluded_transcripts, chomp(line))
                end
            end
        end

        sample = RNASeqSample(parsed_args["transcripts_filename"],
                              parsed_args["genome_filename"],
                              parsed_args["reads_filename"],
                              excluded_seqs,
                              excluded_transcripts,
                              Nullable{String}(parsed_args["output"]))
        return

    elseif subcmd == "likelihood-approx"
        @add_arg_table arg_settings begin
            "--output", "-o"
                default = "sample-data.h5"
            "--approx-method"
                default = "logit_skew_normal_hsb"
            "--tree-method"
                default = "cluster"
            "likelihood_matrix_filename"
                required = true
        end
        parsed_args = parse_args(subcmd_args, arg_settings)

        tree_method = Symbol(parsed_args["tree-method"])
        approx = select_approx_method(parsed_args["approx-method"], tree_method)
        approximate_likelihood(approx,
                               parsed_args["likelihood_matrix_filename"],
                               parsed_args["output"])
        return


    elseif subcmd == "prepare-experiment" || subcmd == "prep"
        @add_arg_table arg_settings begin
            "experiment"
                required = true
            "--force"
                action = :store_true
            "--num-threads", "-t" # handled by the wrapper script
        end

        parsed_args = parse_args(subcmd_args, arg_settings)
        force_overwrite = parsed_args["force"]
        spec = YAML.load_file(parsed_args["experiment"])

        excluded_transcripts = Set{String}() # TODO: read this
        excluded_seqs = Set{String}() # TODO: read this
        no_bias = Bool(get(spec, "no_bias", false))

        approximation_method_name =
            get(spec, "approximation", "logit_skew_normal_hsb")
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
        else
            error(
                """
                Either 'transcripts' or 'genome' and 'annotations' must be specified
                in the experiment specification.
                """)
        end

        if !haskey(spec, "samples")
            warn("No samples specified the experiment specification.")
            exit()
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
                reads_input, reads_cmd_proc =
                    open(Cmd(vcat(reads_decompress_cmd, reads_file)))
            else
                reads_input = reads_file
            end

            sample = RNASeqSample(
                ts, ts_metadata, reads_input,
                excluded_seqs, excluded_transcripts,
                no_bias=no_bias)

            approximate_likelihood(approximation, sample, output_file)
        end


    elseif subcmd == "prepare-sample"
        @add_arg_table arg_settings begin
            "--output", "-o"
                default = "sample-data.h5"
            "transcripts_filename"
                required = true
            "genome_filename"
                required = true
            "reads_filename"
                required = true
            "--exclude-seqs"
                required = false
            "--exclude-transcripts"
                required = false
            "--likelihood-matrix"
                required = false
            "--approx-method"
                default = "logit_skew_normal_hsb"
            "--tree-method"
                default = "cluster"
            "--num-threads", "-t" # handled by the wrapper script
            "--no-bias"
                action = :store_true
            "--dump-bias-training-examples"
                action = :store_true
            "--no-gpu"            # handled by the wrapper script
        end
        parsed_args = parse_args(subcmd_args, arg_settings)

        tree_method = Symbol(parsed_args["tree-method"])
        approx = select_approx_method(parsed_args["approx-method"], tree_method)

        excluded_seqs = Set{String}()
        if parsed_args["exclude-seqs"] != nothing
            open(parsed_args["exclude-seqs"]) do input
                for line in eachline(input)
                    push!(excluded_seqs, chomp(line))
                end
            end
        end

        excluded_transcripts = Set{String}()
        if parsed_args["exclude-transcripts"] != nothing
            open(parsed_args["exclude-transcripts"]) do input
                for line in eachline(input)
                    push!(excluded_transcripts, chomp(line))
                end
            end
        end

        sample = RNASeqSample(
            parsed_args["transcripts_filename"],
            parsed_args["genome_filename"],
            parsed_args["reads_filename"],
            excluded_seqs,
            excluded_transcripts,
            parsed_args["likelihood-matrix"] == nothing ?
              Nullable{String}() :
              Nullable(parsed_args["likelihood-matrix"]),
            no_bias=parsed_args["no-bias"],
            dump_bias_training_examples=parsed_args["dump-bias-training-examples"])
        approximate_likelihood(approx, sample, parsed_args["output"])
        return


    elseif subcmd == "trans-prep"
        @add_arg_table arg_settings begin
            "--output", "-o"
                default = "sample-data.h5"
            "transcript_sequence_filename"
                required = true
            "reads_filename"
                required = true
            "--exclude-seqs"
                required = false
            "--exclude-transcripts"
                required = false
            "--likelihood-matrix"
                required = false
            "--approx-method"
                default = "logit_skew_normal_hsb"
            "--tree-method"
                default = "cluster"
            "--num-threads", "-t" # handled by the wrapper script
            "--no-bias"
                action = :store_true
            "--no-gpu"            # handled by the wrapper script
        end
        parsed_args = parse_args(subcmd_args, arg_settings)

        tree_method = Symbol(parsed_args["tree-method"])
        approx = select_approx_method(parsed_args["approx-method"], tree_method)

        excluded_seqs = Set{String}()
        if parsed_args["exclude-seqs"] != nothing
            open(parsed_args["exclude-seqs"]) do input
                for line in eachline(input)
                    push!(excluded_seqs, chomp(line))
                end
            end
        end

        excluded_transcripts = Set{String}()
        if parsed_args["exclude-transcripts"] != nothing
            open(parsed_args["exclude-transcripts"]) do input
                for line in eachline(input)
                    push!(excluded_transcripts, chomp(line))
                end
            end
        end

        sample = RNASeqSample(parsed_args["transcript_sequence_filename"],
                              parsed_args["reads_filename"],
                              excluded_seqs,
                              excluded_transcripts,
                              parsed_args["likelihood-matrix"] == nothing ?
                                Nullable{String}() :
                                Nullable(parsed_args["likelihood-matrix"]),
                              no_bias=parsed_args["no-bias"])
        approximate_likelihood(approx, sample, parsed_args["output"])
        return

    elseif subcmd == "estimate" || subcmd == "est"
        @add_arg_table arg_settings begin
            "--output", "-o"
                default = Nullable{String}()
            "--output-format", "-F"
                default = "csv"
            "--exclude-transcripts"
                required = false
            "--flat-prior"
                action = :store_true
            "--credible-lower"
                default = 0.025
                arg_type = Float64
            "--credible-upper"
                default = 0.975
                arg_type = Float64
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

        ts, ts_metadata = Transcripts(transcripts_filename, excluded_transcripts)
        gene_db = write_transcripts("genes.db", ts, ts_metadata)

        if parsed_args["flat-prior"]
            global INFORMATIVE_PRIOR
            INFORMATIVE_PRIOR = false
        end

        max_num_samples =  get(parsed_args, "max-num-samples", nothing)
        loaded_samples = load_samples_from_specification(
            spec, ts, ts_metadata, max_num_samples,
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
            credible_interval)

        POLEE_MODELS[parsed_args["model"]](input)

        # TODO: figure out what to do with `output`

        return

    elseif subcmd == "optimize"
        @add_arg_table arg_settings begin
            "--output", "-o"
                default = "transcript-expression-estimates.csv"
            "likelihood_matrix"
                required = true
        end
        parsed_args = parse_args(subcmd_args, arg_settings)
        expectation_maximization(
            parsed_args["likelihood_matrix"], parsed_args["output"])
        return

    elseif subcmd == "sample"
        @add_arg_table arg_settings begin
            "--output", "-o"
                default = "samples.csv"
            "likelihood_matrix"
                required = true
        end
        parsed_args = parse_args(subcmd_args, arg_settings)
        gibbs_sampler(parsed_args["likelihood_matrix"],
                      parsed_args["output"])
        return
    elseif subcmd == "approx-sample"
        @add_arg_table arg_settings begin
            "--output", "-o"
                default = "post_mean.csv"
            "--exclude-transcripts"
                required = false
            "--num-samples"
                default = 1000
                arg_type = Int
            "transcripts"
                required = true
            "prepared_sample"
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

        ts, ts_metadata = Transcripts(parsed_args["transcripts"], excluded_transcripts)
        n = length(ts)

        input = h5open(parsed_args["prepared_sample"])

        input_metadata = g_open(input, "metadata")
        check_prepared_sample_version(input_metadata)
        close(input_metadata)

        node_parent_idxs = read(input["node_parent_idxs"])
        node_js          = read(input["node_js"])
        efflens          = read(input["effective_lengths"])

        mu    = read(input["mu"])
        sigma = exp.(read(input["omega"]))
        alpha = read(input["alpha"])

        t = HSBTransform(node_parent_idxs, node_js)

        num_samples = parsed_args["num-samples"]
        samples = Array{Float32}((num_samples, n))

        zs0 = Array{Float32}(n-1)
        zs = Array{Float32}(n-1)
        ys = Array{Float64}(n-1)
        xs = Array{Float32}(n)

        prog = Progress(num_samples, 0.25, "Sampling from approx. likelihood ", 60)
        for i in 1:num_samples
            next!(prog)
            for j in 1:n-1
                zs0[j] = randn(Float32)
            end
            sinh_asinh_transform!(alpha, zs0, zs, Val{true})
            logit_normal_transform!(mu, sigma, zs, ys, Val{true})
            hsb_transform!(t, ys, xs, Val{true})

            # effective length transform
            xs ./= efflens
            xs ./= sum(xs)
            samples[i, :] = xs
        end
        finish!(prog)

        # TODO: only interested in the mean for now, but we may want to dump
        # all the samples some time.
        post_mean = mean(samples, 1)

        open(parsed_args["output"], "w") do output
            for (j, t) in enumerate(ts)
                println(output, t.metadata.name, ",", post_mean[j])
            end
        end

        return
    else
        println("Unknown command: ", subcmd, "\n")
        print_usage()
        exit(1)
    end
end
