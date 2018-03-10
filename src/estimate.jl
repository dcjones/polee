unshift!(PyVector(pyimport("sys")["path"]), Pkg.dir("Extruder", "src"))
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
@pyimport rnaseq_approx_likelihood


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
    num_samples = length(filenames)
    n = length(ts)

    efflen_values = Array{Float32}(num_samples, n)
    x0_values     = Array{Float32}(num_samples, n)

    # likelihood approximation base distribution parameters
    la_param_values = Array{Float32}(num_samples, 3, n-1)

    # index parameters used to compute inverse heirarchical stick breaking transform
    # leaf_indexes_values                = Array{Int32}(num_samples, n, 2)
    # left_child_rightmost_index  = Array{Int32}(num_samples, n-1, 2)
    # left_child_leftmost_index   = Array{Int32}(num_samples, n-1, 2)
    # right_child_rightmost_index = Array{Int32}(num_samples, n-1, 2)
    # right_child_leftmost_index  = Array{Int32}(num_samples, n-1, 2)

    left_index_values  = Array{Int32}(num_samples, 2*n-1)
    right_index_values = Array{Int32}(num_samples, 2*n-1)
    leaf_index_values  = Array{Int32}(num_samples, 2*n-1)

    prog = Progress(length(filenames), 0.25, "Reading sample data ", 60)
    for (i, filename) in enumerate(filenames)
        input = h5open(filename)

        sample_n = read(input["n"])
        if sample_n != n
            error("Prepared sample has a different number of transcripts than provided GFF3 file.")
        end

        input_metadata = g_open(input, "metadata")
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

        # invhsb_params = make_inverse_hsb_params(node_parent_idxs, node_js)
        # (leafindex, internal_node_left_indexes, internal_node_right_indexes,
        #  leftmost, rightmost) = invhsb_params

        # # make 0-based for python/tensorflow
        # leafindex .-= Int32(1)

        # # make 1-based exclusive
        # leftmost .-= Int32(1)

        # idxs = fill(Int32(i-1), n)
        # leaf_indexes_values[i, :, :] = hcat(idxs, leafindex)

        # idxs = fill(Int32(i-1), n-1)
        # left_child_rightmost_index[i, :, :] = hcat(idxs, rightmost[internal_node_left_indexes])
        # left_child_leftmost_index[i, :, :]  = hcat(idxs, leftmost[internal_node_left_indexes])

        # right_child_rightmost_index[i, :, :] = hcat(idxs, rightmost[internal_node_right_indexes])
        # right_child_leftmost_index[i, :, :]  = hcat(idxs, leftmost[internal_node_right_indexes])

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

        close(input)
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


struct ModelInput
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

    inference[:initialize](n_iter=n_iter, optimizer=optimizer)
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

