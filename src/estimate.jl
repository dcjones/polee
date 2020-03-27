
using LinearAlgebra: I
using Statistics


"""
Read file names and factor annotations from a specification.
"""
function read_specification(
        spec; point_estimates_key=nothing, max_num_samples=nothing)

    prep_file_suffix = get(spec, "prep_file_suffix", ".likelihood.h5")

    sample_names = String[]
    filenames = String[]
    sample_factors = Vector{Dict{String, String}}()
    for sample in spec["samples"]
        sample_name = sample["name"]
        push!(sample_names, sample_name)

        if point_estimates_key === nothing
            push!(filenames,
                get(sample, "file", string(sample["name"], prep_file_suffix)))
        else
            if !haskey(sample, "point-estimates")
                error("Sample $(sample_name) has no point estimate files psecified.")
            end

            if !haskey(sample["point-estimates"], point_estimates_key)
                error("""
                    Sample $(sample_name) has no point estimates specified
                    with key $(point_estimates_key)
                    """)
            end

            push!(filenames, sample["point-estimates"][point_estimates_key])
        end
        factors = Dict{String, String}()
        for (factor_type, factor) in get(sample, "factors", Dict{String, String}())
            factors[factor_type] = string(factor)
        end
        push!(sample_factors, factors)
    end

    num_samples = length(filenames)
    println("Read model specification with ", num_samples, " samples")

    if max_num_samples !== nothing && max_num_samples < num_samples
        max_num_samples = min(max_num_samples, num_samples)
        println("Choosing random subset of ", max_num_samples, " samples")
        p = shuffle(1:num_samples)[1:max_num_samples]
        filenames = filenames[p]
        sample_names = sample_names[p]
        sample_factors = sample_factors[p]
        num_samples = max_num_samples
    end

    return filenames, sample_names, sample_factors
end


"""
Convert a vector of kallisto count estimates into a vector of proportions,
using the set and ordering in transcript_idxs
"""
function kallisto_counts_to_proportions(counts, efflens, pseudocount, transcript_ids, transcript_idx)
    # expand to same set and ordering as ts
    n = length(transcript_idx)
    xs = zeros(Float32, (1,n))
    for (id, count, efflen) in zip(transcript_ids, counts, efflens)
        if haskey(transcript_idx, id)
            xs[1,transcript_idx[id]] = count / efflen
        end
    end

    # convert to proportions
    xs ./= sum(xs)
    xs .+= pseudocount / 1f6
end


"""
Load point estimates from kallisto files. If use bootstrap is true, standard
deviation is estimated from the bootstrap samples.
"""
function load_kallisto_estimates_from_specification(
        spec, ts, ts_metadata, pseudocount, use_bootstrap)

    _, sample_names, sample_factors = read_specification(spec)
    filenames = [entry["kallisto"] for entry in spec["samples"]]

    transcript_idx = Dict{String, Int}()
    for (j, t) in enumerate(ts)
        transcript_idx[t.metadata.name] = j
    end
    n = length(transcript_idx)

    xss = Array{Float32, 2}[]
    xss_log_stds = Array{Float32, 2}[]

    for filename in filenames
        h5open(filename) do input
            transcript_ids = read(input["aux"]["ids"])

            efflens = read(input["aux"]["eff_lengths"])
            est_counts = Vector{Float32}(read(input["est_counts"]))
            xs = kallisto_counts_to_proportions(
                est_counts, efflens, pseudocount, transcript_ids, transcript_idx)

            if use_bootstrap
                bss = Array{Float32, 2}[]
                for dataset in input["bootstrap"]
                    bootstrap_counts = Vector{Float32}(read(dataset))
                    bs = kallisto_counts_to_proportions(
                        bootstrap_counts, efflens, pseudocount, transcript_ids,
                        transcript_idx)
                    push!(bss, bs)
                end

                log_bs = log.(vcat(bss...))
                push!(xss_log_stds, max.(1e-2, std(log_bs, dims=1)))
                push!(xss, exp.(mean(log_bs, dims=1)))
            else
                push!(xss, xs)
            end
        end
    end

    ls = LoadedSamples(
        Array{Float32}(undef, (0,0)),
        vcat(xss...),
        use_bootstrap ? vcat(xss_log_stds...) : nothing,
        Array{Float32}(undef, (0,0)),
        Array{Float32}(undef, (0,0)),
        Array{Float32}(undef, (0,0)),
        Array{Int32}(undef, (0,0)),
        Array{Int32}(undef, (0,0)),
        Array{Int32}(undef, (0,0)),
        Dict{String, Any}(),
        Dict{Any, Any}(),
        sample_factors,
        sample_names,
        filenames)

    return ls
end


"""
Read point estimates from csv files (rather than approximated likelihood
in hdf5 files)

"""
function load_point_estimates_from_specification(
        spec, ts, ts_metadata, point_estimates_key;
        max_num_samples=nothing)

    filenames, sample_names, sample_factors =
        read_specification(
            spec,
            point_estimates_key=point_estimates_key,
            max_num_samples=max_num_samples)

    loaded_samples = load_point_estimates(
        filenames, ts, ts_metadata)

    loaded_samples.sample_factors = sample_factors
    loaded_samples.sample_names = sample_names

    return loaded_samples
end


"""
Read point estimates from csv files (rather than approximated likelihood
in hdf5 files)

"""
function load_samplers_from_specification(spec, ts, ts_metadata; max_num_samples=nothing)

    filenames, sample_names, sample_factors =
        read_specification(
            spec,
            max_num_samples=max_num_samples)

    num_samples = length(filenames)
    n = length(ts)

    samplers = Polee.ApproxLikelihoodSampler[]
    efflens = Array{Float32}(undef, (num_samples, n))

    for (i, filename) in enumerate(filenames)
        input = h5open(filename)

        node_parent_idxs = read(input["node_parent_idxs"])
        node_js          = read(input["node_js"])
        efflens_i        = read(input["effective_lengths"])
        m                = read(input["m"])

        mu    = read(input["mu"])
        sigma = exp.(read(input["omega"]))
        alpha = read(input["alpha"])

        t = Polee.PolyaTreeTransform(node_parent_idxs, node_js)

        sampler = Polee.ApproxLikelihoodSampler()
        Polee.set_transform!(sampler, t, mu, sigma, alpha)
        push!(samplers, sampler)

        efflens[i,:] = efflens_i
    end

    return samplers, efflens, sample_names, sample_factors
end


"""
Load samples from a a YAML specification file.

Input:
    * experiment_spec_filename: filename of YAML experiment specification
    * ts_metadata: transcript metadata
"""
function load_samples_from_specification(
        spec, ts, ts_metadata;
        max_num_samples=nothing, batch_size=nothing,
        check_gff_hash::Bool=true,
        using_tensorflow::Bool=true)

    filenames, sample_names, sample_factors =
        read_specification(spec, max_num_samples=max_num_samples)

    num_samples = length(filenames)

    batch_size = batch_size === nothing ?
        num_samples : min(batch_size, num_samples)

    loaded_samples = load_samples(
        filenames, ts, ts_metadata, batch_size,
        check_gff_hash=check_gff_hash,
        using_tensorflow=using_tensorflow)
    println("Sample data loaded")

    loaded_samples.sample_factors = sample_factors
    loaded_samples.sample_names = sample_names

    return loaded_samples
end


function read_transcripts_filename_from_prepared(filename)
    input = h5open(filename)
    input_metadata = g_open(input, "metadata")
    transcripts_filename = read(attrs(input_metadata)["gfffilename"])
    close(input_metadata)
    close(input)
    return transcripts_filename
end


"""
Load sample point estimates from CSV files.
"""
function load_point_estimates(filenames, ts, ts_metadata)
    num_samples = length(filenames)
    n = length(ts)

    x0_values = zeros(Float32, (num_samples, n))

    transcript_idx = Dict{String, Int}()
    for (j, t) in enumerate(ts)
        transcript_idx[t.metadata.name] = j
    end

    prog = Progress(length(filenames), 0.25, "Reading sample data ", 60)
    for (i, filename) in enumerate(filenames)
        open(filename) do input
            header = split(readline(input), ',')
            @assert header[1] == "transcript_id"
            @assert header[2] == "tpm"

            for line in eachline(input)
                row = split(line, ',')
                if haskey(transcript_idx, row[1])
                    j = transcript_idx[row[1]]
                    x0_values[i, j] = parse(Float64, row[2])/1f6
                else
                    # @warn "transcript not present in annotations" row[1]
                end
            end
        end

        next!(prog)
    end
    println("Sample data loaded")

    return LoadedSamples(
        Array{Float32}(undef, (0,0)),
        x0_values,
        nothing,
        Array{Float32}(undef, (0,0)),
        Array{Float32}(undef, (0,0)),
        Array{Float32}(undef, (0,0)),
        Array{Int32}(undef, (0,0)),
        Array{Int32}(undef, (0,0)),
        Array{Int32}(undef, (0,0)),
        Dict{String, Any}(),
        Dict{Any, Any}(),
        Vector{String}[],
        String[],
        filenames)
end


"""
Load samples from a vector of filenames.

Input:
    * filenames: vector of filenames
    * ts_metadata: transcript metadata
"""
function load_samples(
        filenames, ts, ts_metadata::TranscriptsMetadata, batch_size;
        check_gff_hash::Bool=true,
        using_tensorflow::Bool=true)
    return load_samples_hdf5(
        filenames, ts, ts_metadata, batch_size,
        check_gff_hash=check_gff_hash,
        using_tensorflow=using_tensorflow)
end


function load_samples_hdf5(
        filenames, ts, ts_metadata::TranscriptsMetadata, batch_size;
        check_gff_hash::Bool=true,
        using_tensorflow::Bool=true)
    num_samples = length(filenames)
    n = length(ts)

    efflen_values = Array{Float32}(undef, (num_samples, n))
    x0_values     = zeros(Float32, (num_samples, n))

    # likelihood approximation base distribution parameters
    la_mu_values    = Array{Float32}(undef, (num_samples, n-1))
    la_sigma_values = Array{Float32}(undef, (num_samples, n-1))
    la_alpha_values = Array{Float32}(undef, (num_samples, n-1))

    left_index_values  = Array{Int32}(undef, (num_samples, 2*n-1))
    right_index_values = Array{Int32}(undef, (num_samples, 2*n-1))
    leaf_index_values  = Array{Int32}(undef, (num_samples, 2*n-1))

    mu    = Array{Float32}(undef, n-1)
    sigma = Array{Float32}(undef, n-1)
    alpha = Array{Float32}(undef, n-1)
    node_parent_idxs = Array{Int32}(undef, 2*n-1)
    node_js          = Array{Int32}(undef, 2*n-1)
    efflen_values_i = Array{Float32}(undef, n)

    prog = Progress(length(filenames), 0.25, "Reading sample data ", 60)
    for (i, filename) in enumerate(filenames)
        input = h5open(filename)

        sample_n = read(input["n"])
        if sample_n != n
            error("Prepared sample $(filename) has a different number of transcripts than provided GFF3 file.")
        end

        input_metadata = g_open(input, "metadata")

        Polee.check_prepared_sample_version(input_metadata)

        if check_gff_hash && base64decode(read(attrs(input_metadata)["gffhash"])) != ts_metadata.gffhash
            true_filename = read(attrs(input_metadata)["gfffilename"])
            error(
                """
                $(filename):
                GFF3 file is not the same as the one used for sample preparation.
                Filename of original GFF3 file: $(true_filename)
                """)
        end
        close(input_metadata)

        # TODO: record and check transcript blacklist hash.

        HDF5.readarray(input["mu"], HDF5.hdf5_type_id(Float32), mu)
        HDF5.readarray(input["omega"], HDF5.hdf5_type_id(Float32), sigma)
        map!(exp, sigma, sigma)
        HDF5.readarray(input["alpha"], HDF5.hdf5_type_id(Float32), alpha)

        la_mu_values[i, :]    = mu
        la_sigma_values[i, :] = sigma
        la_alpha_values[i, :] = alpha

        HDF5.readarray(input["effective_lengths"], HDF5.hdf5_type_id(Float32), efflen_values_i)
        efflen_values[i, :] = efflen_values_i

        HDF5.readarray(input["node_parent_idxs"], HDF5.hdf5_type_id(Int32), node_parent_idxs)
        HDF5.readarray(input["node_js"], HDF5.hdf5_type_id(Int32), node_js)

        close(input)

        left_index, right_index, leaf_index =
            Polee.make_inverse_ptt_params(node_parent_idxs, node_js)

        left_index_values[i, :]  = left_index
        right_index_values[i, :] = right_index
        leaf_index_values[i, :]  = leaf_index

        # find reasonable initial values by estimating the mean
        N = 30
        y0 = Array{Float64}(undef, n-1)
        x0 = Array{Float32}(undef, n)
        t = Polee.PolyaTreeTransform(node_parent_idxs, node_js)
        for _ in 1:N
            for j in 1:n-1
                z0 = randn(Float32)
                z = sinh(asinh(z0) + alpha[j])
                y0[j] = clamp(
                    Polee.logistic(mu[j] + z * sigma[j]),
                    Polee.LIKAP_Y_EPS, 1 - Polee.LIKAP_Y_EPS)
            end
            Polee.transform!(t, y0, x0)

            x0 ./= efflen_values[i,:]
            x0 ./= sum(x0)
            x0_values[i, :] .+= x0
        end
        x0_values[i,:] ./= N

        next!(prog)
    end

    var_names = [
        "efflen",
        "la_mu",
        "la_sigma",
        "la_alpha",
        "left_index",
        "right_index",
        "leaf_index"]

    var_values = Any[
        efflen_values,
        la_mu_values,
        la_sigma_values,
        la_alpha_values,
        left_index_values,
        right_index_values,
        leaf_index_values]

    ls = LoadedSamples(
        efflen_values,
        x0_values,
        nothing,
        la_mu_values,
        la_sigma_values,
        la_alpha_values,
        left_index_values,
        right_index_values,
        leaf_index_values,
        Dict{String, Any}(),
        Dict{Any, Any}(),
        Vector{String}[],
        String[],
        filenames)

    if using_tensorflow
        create_tensorflow_variables!(ls, batch_size)
    end

    return ls
end


function create_tensorflow_variables!(ls::LoadedSamples, batch_size=nothing)
    var_names = [
        "efflen",
        "la_mu",
        "la_sigma",
        "la_alpha",
        "left_index",
        "right_index",
        "leaf_index"]

    var_values = Any[
        ls.efflen_values,
        ls.la_mu_values,
        ls.la_sigma_values,
        ls.la_alpha_values,
        ls.left_index,
        ls.right_index,
        ls.leaf_index]

    # Set up tensorflow variables with placeholders to feed likelihood
    # approximation parameters to inference.
    empty!(ls.variables)
    empty!(ls.init_feed_dict)

    for (name, val) in zip(var_names, var_values)
        typ = eltype(val) == Float32 ? tf.float32 : tf.int32
        if batch_size !== nothing
            sz = (batch_size, size(val)[2:end]...)
            # sz = (nothing, size(val)[2:end]...)
        else
            sz = size(val)
        end

        # sz = (nothing, size(val)[2:end]...)

        # var_init = tf.placeholder(typ, shape=sz, name=string(name, "-placeholder"))
        # ls.init_feed_dict[var_init] = val

        # if we are batching, no point in initializing variables
        # the variables only exsit so we can initialize once without
        # having to feed the same data repeatedly.
        # if batch_size !== nothing
        #     ls.variables[name] = var_init
        # else
            var = tf.Variable(val, name=name, trainable=false)
            ls.variables[name] = var
        # end
    end
end


mutable struct ModelInput
    loaded_samples::LoadedSamples
    inference::Symbol
    feature::Symbol
    ts::Transcripts
    ts_metadata::TranscriptsMetadata
    output_filename::Union{String, Nothing}
    output_format::Symbol
    gene_db::SQLite.DB
    credible_interval::Tuple{Float64, Float64}
    parsed_args::Dict{String, Any}
end


"""
Construct a python RNASeqApproxLikelihood class.
"""
function RNASeqApproxLikelihood(input::ModelInput, x)
    return RNASeqApproxLikelihood(input.loaded_samples, x)
end


function RNASeqApproxLikelihood(
        loaded_samples::LoadedSamples, x)
    invhsb_params = [
        loaded_samples.variables[:left_index],
        loaded_samples.variables[:right_index],
        loaded_samples.variables[:leaf_index]
    ]

    @show INFORMATIVE_PRIOR
    return polee_py.RNASeqApproxLikelihood(
            x=x,
            efflens=loaded_samples.variables[:efflen],
            la_params=loaded_samples.variables[:la_param],
            informative_prior=INFORMATIVE_PRIOR,
            invhsb_params=invhsb_params,
            value=Float32[])
end


function rnaseq_approx_likelihood_sampler(input::ModelInput)
    hsb_params = [
        input.loaded_samples.variables[:left_index],
        input.loaded_samples.variables[:right_index],
        input.loaded_samples.variables[:leaf_index]
    ]

    return polee_py.rnaseq_approx_likelihood_sampler(
        efflens=input.loaded_samples.variables[:efflen],
        la_params=input.loaded_samples.variables[:la_param],
        hsb_params=hsb_params)
end


"""
Inference for implicit models. These models models can evaluate the likelihood of
some (not necessarily bijective) transformation of the transcript expression
P(T(x) | theta), but can't necessarily evaluate P(x | theta). Splicing ratios
are the most notable model in this category.

To deal with this, we do stochastic gradient descent by sampling x' from the
likelihood function, computing T(x') and treating that as a new observation
of T(x) at every iteration. This is less efficient mode of inference, but possibly
better than introducing a ton of nusiance parameters to try to evaluate P(x|T(x)).
"""
function run_implicit_model_map_inference(input, Tx, T, latent_vars, n_iter, optimizer)
    sess = ed.get_session()

    x_sample = rnaseq_approx_likelihood_sampler(input)
    Tx_sample = T(x_sample)

    data = Dict(Tx => tf.placeholder(tf.float32, shape=Tx[:get_shape]()))

    inference = ed.MAP(latent_vars=latent_vars, data=data)

    # inference[:initialize](n_iter=n_iter, optimizer=optimizer, n_print=1, logdir="log")
    # inference[:initialize](n_iter=n_iter, optimizer=optimizer, logdir="log")
    inference[:initialize](n_iter=n_iter, optimizer=optimizer)

    sess[:run](tf.global_variables_initializer(),
               feed_dict=input.loaded_samples.init_feed_dict)

    # run_options = tf.RunOptions(trace_level=tf.RunOptions[:FULL_TRACE])
    # run_metadata = tf.RunMetadata()

    # smpl = sess[:run](Tx_sample, feed_dict=input.loaded_samples.init_feed_dict)
    # feed_dict = merge(input.loaded_samples.init_feed_dict, Dict(data[Tx] => smpl))
    # sess[:run]([inference[:train], inference[:increment_t], inference[:loss]],
    #            options=run_options, run_metadata=run_metadata,
    #            feed_dict=feed_dict)
    # tl = tftl.Timeline(run_metadata[:step_stats])
    # ctf = tl[:generate_chrome_trace_format]()
    # trace_out = pybuiltin(:open)("timeline.json", "w")
    # trace_out[:write](ctf)
    # trace_out[:close]()
    # exit()

    for iter in 1:n_iter
        try
            feed_dict = Dict(data[Tx] =>
                sess[:run](Tx_sample, feed_dict=input.loaded_samples.init_feed_dict))
            # info_dict = inference[:update](merge(feed_dict, input.loaded_samples.init_feed_dict))
            info_dict = inference[:update](feed_dict)
            inference[:print_progress](info_dict)
        catch ex
            @show ex
            @show ex.val[:message]
            rethrow(ex)
        end
    end

    inference[:finalize]()
end


"""
Aide optimization by holding some variables fixed for some number initial of iterations.
"""
function run_two_stage_klqp_inference(input, latent_variables, frozen_variables,
                                      frozen_variables_vals, data,
                                      n_iter1, n_iter2, optimizer)
    sess = ed.get_session()

    # stage 1
    pre_opt_latent_variables = copy(latent_variables)
    for (var, val) in zip(frozen_variables, frozen_variables_vals)
        pre_opt_latent_variables[var] =
            edmodels.Normal(
                loc=tf.fill(var[:get_shape](), val),
                scale=tf.constant([0.0001f0]), name="Stage1SemiFixedNormal")
            # edmodels.PointMass(tf.fill(var[:get_shape](), val))

        # or, just exclude it...
        # delete!(pre_opt_latent_variables, var)
    end

    pre_opt_inference = ed.KLqp(pre_opt_latent_variables, data=data)
    pre_opt_inference[:initialize](n_iter=n_iter1, optimizer=optimizer,
                                   auto_transform=false, logdir="log")
    sess[:run](tf.global_variables_initializer(),
            feed_dict=input.loaded_samples.init_feed_dict)

    for iter in 1:n_iter1
        info_dict = pre_opt_inference[:update]()
        pre_opt_inference[:print_progress](info_dict)
    end
    pre_opt_inference[:finalize]()

    # stage 2
    inference = ed.KLqp(latent_variables, data=data)
    inference[:initialize](n_iter=n_iter2, optimizer=optimizer,
                           auto_transform=false, logdir="log")

    uninitialized_vars = Any[]
    for rv in frozen_variables
        for var in latent_variables[rv][:get_variables]()
            if pyisinstance(var, tf.Variable)
                push!(uninitialized_vars, var)
            end
        end
    end
    sess[:run](tf.variables_initializer(uninitialized_vars),
               feed_dict=input.loaded_samples.init_feed_dict)
    sess[:run](inference[:reset])

    for iter in 1:n_iter2
        info_dict = inference[:update]()
        inference[:print_progress](info_dict)
    end
    inference[:finalize]()
end


"""
This is essentiall Inference.run from edward, but customized to avoid
reinitializing variables on each iteration.
"""
function run_inference(input::ModelInput, inference, n_iter, optimizer)
    return run_inference(input.loaded_samples.init_feed_dict, inference, n_iter, optimizer)
end


function run_inference(init_feed_dict::Dict, inference, n_iter, optimizer)
    sess = ed.get_session()

    # enable XLA
    # config = tf.ConfigProto()
    # config = tf.ConfigProto(device_count = Dict("GPU" => 0))
    # config[:graph_options][:optimizer_options][:global_jit_level] = tf.OptimizerOptions[:ON_1]
    # ed.util[:graphs][:_ED_SESSION] = tf.InteractiveSession(config=config)
    # ed.util[:graphs][:_ED_SESSION] = tf.InteractiveSession()

    inference[:initialize](n_iter=n_iter, optimizer=optimizer)
    # inference[:initialize](n_iter=n_iter, optimizer=optimizer, auto_transform=false, logdir="log")
    # inference[:initialize](n_iter=n_iter)

    sess[:run](tf.global_variables_initializer(), feed_dict=init_feed_dict)

    for iter in 1:n_iter
        local info_dict
        try
            info_dict = inference[:update]()
        catch e
            @show e.T
            @show e.msg
            @show e.val["message"]
            # @show typeof(e)
            # @show fieldnames(e)
            throw()
        end
        inference[:print_progress](info_dict)
    end

    # profiler
    # --------
    # prof_opt_builder = tf.profiler[:ProfileOptionBuilder]
    # prof_opts = prof_opt_builder(prof_opt_builder[:time_and_memory]())[:order_by]("micros")[:build]()

    # run_options = tf.RunOptions(trace_level=tf.RunOptions[:FULL_TRACE])
    # run_metadata = tf.RunMetadata()
    # sess[:run]([inference[:train], inference[:increment_t], inference[:loss]],
    #            options=run_options, run_metadata=run_metadata)

    # tf.profiler[:profile](sess[:graph], run_meta=run_metadata, options=prof_opts)

    # timeline profiler
    # -----------------
    # run_options = tf.RunOptions(trace_level=tf.RunOptions[:FULL_TRACE])
    # run_metadata = tf.RunMetadata()
    # sess[:run]([inference[:train], inference[:increment_t], inference[:loss]],
    #            options=run_options, run_metadata=run_metadata)

    # tl = tftl.Timeline(run_metadata[:step_stats])
    # ctf = tl[:generate_chrome_trace_format]()
    # trace_out = pybuiltin(:open)("timeline.json", "w")
    # trace_out[:write](ctf)
    # trace_out[:close]()

    inference[:finalize]()
end


function reset_graph()
    tf.reset_default_graph()
    old_sess = ed.get_session()
    old_sess[:close]()
    ed.util[:graphs][:_ED_SESSION] = tf.InteractiveSession()
end


function write_effects_csv(filename, factoridx, W)
    n = size(W, 2)
    open(filename, "w") do output
        println(output, "factor,id,w")
        for factor = sort(collect(keys(factoridx)))
            idx = factoridx[factor]
            for j in 1:n
                @printf(output, "%s,%d,%e\n", factor, j, W[idx, j])
            end
        end
    end
end


function write_effects(output_filename, factoridx, W, w_sigma,
                       lower_credible, upper_credible, error_sigma, feature)
    println("Writing regression results to ", output_filename)

    db = SQLite.DB(output_filename)

    SQLite.execute!(db, "drop table if exists effects")
    SQLite.execute!(db,
        """
        create table effects
        ($(feature)_num INT, factor TEXT, w REAL, w_sigma REAL, w_lower REAL, w_upper REAL, error_sigma REAL)
        """)

    ins_stmt = SQLite.Stmt(db, "insert into effects values (?1, ?2, ?3, ?4, ?5, ?6, ?7)")

    SQLite.execute!(db, "begin transaction")
    n = size(W, 2)
    for factor = sort(collect(keys(factoridx)))
        idx = factoridx[factor]
        for j in 1:n
            SQLite.bind!(ins_stmt, 1, j)
            SQLite.bind!(ins_stmt, 2, factor)
            SQLite.bind!(ins_stmt, 3, W[idx, j])
            SQLite.bind!(ins_stmt, 4, w_sigma[idx, j])
            SQLite.bind!(ins_stmt, 5, lower_credible[idx, j])
            SQLite.bind!(ins_stmt, 6, upper_credible[idx, j])
            SQLite.bind!(ins_stmt, 7, error_sigma[j])
            SQLite.execute!(ins_stmt)
        end
    end
    SQLite.execute!(db, "end transaction")
end


function write_estimates(filename, names, est)
    n = size(est, 2)
    open(filename, "w") do output
        # TODO: where do we get the transcript names from?
        println(output, "name,id,tpm")
        for (i, name) in enumerate(names)
            for j in 1:n
                @printf(output, "%s,%d,%e\n", name, j, est[i, j])
            end
        end
    end
end

