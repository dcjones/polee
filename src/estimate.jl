

mutable struct LoadedSamples
    # effective lengths
    efflen_values::Array{Float32, 2}

    # reasonable initial values
    x0_values::Array{Float32, 2}

    # likelihood approximation base distribution parameters
    la_param_values::Array{Float32, 3}

    # hsb parameters
    left_index::Array{Int32, 2}
    right_index::Array{Int32, 2}
    leaf_index::Array{Int32, 2}

    # tensorflow placeholder variables and initalizations
    # variables::Dict{Symbol, PyObject}
    variables::Dict{Symbol, Any}
    init_feed_dict::Dict{Any, Any}

    sample_factors::Vector{Vector{String}}
    sample_names::Vector{String}
end


"""
Load samples from a a YAML specification file.

Input:
    * experiment_spec_filename: filename of YAML experiment specification
    * ts_metadata: transcript metadata
"""
function load_samples_from_specification(experiment_spec_filename, ts, ts_metadata)
    experiment_spec = YAML.load_file(experiment_spec_filename)
    sample_names = [entry["name"] for entry in experiment_spec]
    filenames = [entry["file"] for entry in experiment_spec]
    sample_factors = [get(entry, "factors", String[]) for entry in experiment_spec]
    num_samples = length(filenames)
    println("Read model specification with ", num_samples, " samples")

    loaded_samples = load_samples(filenames, ts, ts_metadata)
    println("Sample data loaded")

    loaded_samples.sample_factors = sample_factors
    loaded_samples.sample_names = sample_names

    return loaded_samples
end


"""
Load samples from a vector of filenames.

Input:
    * filenames: vector of filenames
    * ts_metadata: transcript metadata
"""
function load_samples(filenames, ts, ts_metadata::TranscriptsMetadata)
    return load_samples_hdf5(filenames, ts, ts_metadata)
end


function load_samples_flatbuffer(filenames, ts, ts_metadata::TranscriptsMetadata)
    num_samples = length(filenames)
    n = length(ts)

    efflen_values = Array{Float32}(num_samples, n)
    x0_values     = Array{Float32}(num_samples, n)

    # likelihood approximation base distribution parameters
    la_param_values = Array{Float32}(num_samples, 3, n-1)

    left_index_values  = Array{Int32}(num_samples, 2*n-1)
    right_index_values = Array{Int32}(num_samples, 2*n-1)
    leaf_index_values  = Array{Int32}(num_samples, 2*n-1)

    mut = Threads.Mutex()

    prog = Progress(length(filenames), 0.25, "Reading sample data ", 60)

    Threads.@threads for i in 1:length(filenames)
    # for i in 1:length(filenames)
        filename = filenames[i]
        data = read_approximated_likelihood_data(filename)

        if length(data.mu) + 1 != n
            error("Prepared sample has a different number of transcripts than provided GFF3 file.")
        end

        if data.gff_hash != ts_metadata.gffhash
            error(
                """
                $(filename):
                GFF3 file is not the same as the one used for sample preparation.
                Filename of original GFF3 file: $(data.gff_filename)
                """)
        end

        # TODO: record and check transcript blacklist hash.

        sigma = exp.(data.omega)

        left_index, right_index, leaf_index =
            make_inverse_hsb_params(data.parent_idxs, data.leaf_idxs)

        # find reasonable initial valuse by taking the mean of the base normal
        # distribution and transforming it
        y0 = Array{Float64}(n-1)
        x0 = Array{Float32}(n)
        for j in 1:n-1
            y0[j] = clamp(logistic(sinh(data.alpha[j]) + data.mu[j]), LIKAP_Y_EPS, 1 - LIKAP_Y_EPS)
        end
        t = HSBTransform(data.parent_idxs, data.leaf_idxs)
        hsb_transform!(t, y0, x0, Val{true})

        lock(mut)
        la_param_values[i, 1, :] = data.mu
        la_param_values[i, 2, :] = sigma
        la_param_values[i, 3, :] = data.alpha

        efflen_values[i, :] = data.efflens

        left_index_values[i, :]  = left_index
        right_index_values[i, :] = right_index
        leaf_index_values[i, :]  = leaf_index

        x0_values[i, :] = x0
        next!(prog)
        unlock(mut)
    end

    # this allocates a great deal very quickly which may not be dealt with in time
    gc()

    var_names = [:la_param,
                 :efflen,
                 :left_index,
                 :right_index,
                 :leaf_index]
    var_values = Any[la_param_values,
                     efflen_values,
                     left_index_values,
                     right_index_values,
                     leaf_index_values]

    variables = Dict{Symbol, PyObject}()
    init_feed_dict = Dict{Any, Any}()
    for (name, val) in zip(var_names, var_values)
        typ = eltype(val) == Float32 ? tf.float32 : tf.int32
        var_init = tf.placeholder(typ, shape=size(val))
        var      = tf.Variable(var_init)
        variables[name] = var
        init_feed_dict[var_init] = val
    end

    return LoadedSamples(
        efflen_values,
        x0_values,
        la_param_values,
        left_index_values,
        right_index_values,
        leaf_index_values,
        variables,
        init_feed_dict,
        Vector{String}[], String[])
end


function load_samples_hdf5(filenames, ts, ts_metadata::TranscriptsMetadata)
    num_samples = length(filenames)
    n = length(ts)

    efflen_values = Array{Float32}(num_samples, n)
    x0_values     = Array{Float32}(num_samples, n)

    # likelihood approximation base distribution parameters
    la_param_values = Array{Float32}(num_samples, 3, n-1)

    left_index_values  = Array{Int32}(num_samples, 2*n-1)
    right_index_values = Array{Int32}(num_samples, 2*n-1)
    leaf_index_values  = Array{Int32}(num_samples, 2*n-1)

    prog = Progress(length(filenames), 0.25, "Reading sample data ", 60)
    for (i, filename) in enumerate(filenames)
        input = h5open(filename)

        sample_n = read(input["n"])
        if sample_n != n
            error("Prepared sample $(filename) has a different number of transcripts than provided GFF3 file.")
        end

        input_metadata = g_open(input, "metadata")

        check_prepared_sample_version(input_metadata)

        if base64decode(read(attrs(input_metadata)["gffhash"])) != ts_metadata.gffhash
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

        mu = read(input["mu"])
        sigma = exp.(read(input["omega"]))
        alpha = read(input["alpha"])

        la_param_values[i, 1, :] = mu
        la_param_values[i, 2, :] = sigma
        la_param_values[i, 3, :] = alpha

        efflen_values[i, :] = read(input["effective_lengths"])

        node_parent_idxs = read(input["node_parent_idxs"])
        node_js = read(input["node_js"])

        close(input)

        left_index, right_index, leaf_index =
            make_inverse_hsb_params(node_parent_idxs, node_js)

        left_index_values[i, :]  = left_index
        right_index_values[i, :] = right_index
        leaf_index_values[i, :]  = leaf_index

        # find reasonable initial valuse by taking the mean of the base normal
        # distribution and transforming it
        y0 = Array{Float64}(n-1)
        x0 = Array{Float32}(n)
        for j in 1:n-1
            y0[j] = clamp(logistic(sinh(alpha[j]) + mu[j]), LIKAP_Y_EPS, 1 - LIKAP_Y_EPS)
        end
        t = HSBTransform(node_parent_idxs, node_js)
        hsb_transform!(t, y0, x0, Val{true})
        x0_values[i, :] = x0

        next!(prog)
    end

    var_names = [:la_param,
                 :efflen,
                 :left_index,
                 :right_index,
                 :leaf_index]
    var_values = Any[la_param_values,
                     efflen_values,
                     left_index_values,
                     right_index_values,
                     leaf_index_values]

    variables = Dict{Symbol, PyObject}()
    init_feed_dict = Dict{Any, Any}()
    for (name, val) in zip(var_names, var_values)
        typ = eltype(val) == Float32 ? tf.float32 : tf.int32
        var_init = tf.placeholder(typ, shape=size(val))
        var      = tf.Variable(var_init)
        variables[name] = var
        init_feed_dict[var_init] = val
    end

    return LoadedSamples(
        efflen_values,
        x0_values,
        la_param_values,
        left_index_values,
        right_index_values,
        leaf_index_values,
        variables,
        init_feed_dict,
        Vector{String}[], String[])
end


mutable struct ModelInput
    loaded_samples::LoadedSamples
    inference::Symbol
    feature::Symbol
    ts::Transcripts
    ts_metadata::TranscriptsMetadata
    output_filename::Nullable{String}
    output_format::Symbol
    gene_db::SQLite.DB
    credible_interval::Tuple{Float64, Float64}
end


"""
Construct a python RNASeqApproxLikelihood class.
"""
function RNASeqApproxLikelihood(input::ModelInput, x)
    invhsb_params = [
        input.loaded_samples.variables[:left_index],
        input.loaded_samples.variables[:right_index],
        input.loaded_samples.variables[:leaf_index]
    ]

    return rnaseq_approx_likelihood.RNASeqApproxLikelihood(
            x=x,
            efflens=input.loaded_samples.variables[:efflen],
            la_params=input.loaded_samples.variables[:la_param],
            invhsb_params=invhsb_params,
            value=Float32[])
end


function rnaseq_approx_likelihood_sampler(input::ModelInput)
    hsb_params = [
        input.loaded_samples.variables[:left_index],
        input.loaded_samples.variables[:right_index],
        input.loaded_samples.variables[:leaf_index]
    ]

    return rnaseq_approx_likelihood.rnaseq_approx_likelihood_sampler(
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

    inference[:initialize](n_iter=n_iter, optimizer=optimizer)

    sess[:run](tf.global_variables_initializer(),
               feed_dict=input.loaded_samples.init_feed_dict)

    for iter in 1:n_iter
        feed_dict = Dict(data[Tx] =>
            sess[:run](Tx_sample, feed_dict=input.loaded_samples.init_feed_dict))
        info_dict = inference[:update](merge(feed_dict, input.loaded_samples.init_feed_dict))
        # info_dict = inference[:update](feed_dict)
        inference[:print_progress](info_dict)
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
function run_inference(input, inference, n_iter, optimizer)
    sess = ed.get_session()

    # enable XLA
    # config = tf.ConfigProto()
    # config = tf.ConfigProto(device_count = Dict("GPU" => 0))
    # config[:graph_options][:optimizer_options][:global_jit_level] = tf.OptimizerOptions[:ON_1]
    # ed.util[:graphs][:_ED_SESSION] = tf.InteractiveSession(config=config)
    # ed.util[:graphs][:_ED_SESSION] = tf.InteractiveSession()

    # inference[:initialize](n_iter=n_iter, optimizer=optimizer)
    inference[:initialize](n_iter=n_iter, optimizer=optimizer, logdir="log")
    # inference[:initialize](n_iter=n_iter)

    sess[:run](tf.global_variables_initializer(),
            feed_dict=input.loaded_samples.init_feed_dict)

    for iter in 1:n_iter
        info_dict = inference[:update]()
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

