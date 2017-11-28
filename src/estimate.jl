unshift!(PyVector(pyimport("sys")["path"]), Pkg.dir("Extruder", "src"))
@pyimport tensorflow as tf
@pyimport tensorflow.contrib.distributions as tfdist
@pyimport tensorflow.python.client.timeline as tftl
@pyimport edward as ed
@pyimport edward.models as edmodels
@pyimport rnaseq_approx_likelihood



"""
Load samples from a a YAML specification file.

Input:
    * experiment_spec_filename: filename of YAML experiment specification
    * ts_metadata: transcript metadata
"""
function load_samples_from_specification(experiment_spec_filename, ts_metadata)
    experiment_spec = YAML.load_file(experiment_spec_filename)
    names = [entry["name"] for entry in experiment_spec]
    filenames = [entry["file"] for entry in experiment_spec]
    sample_factors = [get(entry, "factors", String[]) for entry in experiment_spec]
    num_samples = length(filenames)
    println("Read model specification with ", num_samples, " samples")

    (likapprox_laparam, likapprox_efflen, likapprox_invhsb_params,
     likapprox_parent_idxs, likapprox_js, x0) = load_samples(filenames, ts_metadata)
    println("Sample data loaded")

    return (likapprox_laparam, likapprox_efflen,
            likapprox_invhsb_params, likapprox_parent_idxs, likapprox_js,
            x0, sample_factors, names)
end


"""
Load samples from a vector of filenames.

Input:
    * filenames: vector of filenames
    * ts_metadata: transcript metadata
"""
function load_samples(filenames, ts_metadata::TranscriptsMetadata)
    laparam_tensors = []
    efflen_tensors = []
    x0_tensors = []
    node_parent_idxs_tensors = []
    node_js_tensors = []
    As_tensors = []

    for filename in filenames
        input = h5open(filename, "r")

        n = read(input["n"])

        mu = read(input["mu"])
        sigma = read(input["omega"])
        map!(exp, sigma, sigma)
        alpha = read(input["alpha"])

        node_parent_idxs = read(input["node_parent_idxs"])
        node_js = read(input["node_js"])
        effective_lengths = read(input["effective_lengths"])

        As = inverse_hsb_matrices(node_parent_idxs, node_js)
        push!(As_tensors, As)

        # choose logit-normal mean (actually: not really the mean)
        y0 = Array{Float64}(n-1)
        x0 = Array{Float32}(n)
        for i in 1:n-1
            y0[i] = clamp(logistic(sinh(alpha[i]) + mu[i]), LIKAP_Y_EPS, 1 - LIKAP_Y_EPS)
        end
        t = HSBTransform(node_parent_idxs, node_js)
        hsb_transform!(t, y0, x0, Val{true})
        push!(x0_tensors, x0)

        tf_mu = tf.constant(mu)
        tf_sigma = tf.constant(sigma)
        tf_alpha = tf.constant(alpha)
        tf_laparam = tf.stack([tf_mu, tf_sigma, tf_alpha])
        tf_efflen = tf.constant(effective_lengths)
        push!(laparam_tensors, tf_laparam)
        push!(efflen_tensors, tf_efflen)
        push!(node_parent_idxs_tensors, node_parent_idxs)
        push!(node_js_tensors, node_js)
    end

    return (tf.stack(laparam_tensors), tf.stack(efflen_tensors),
            As_tensors, hcat(node_parent_idxs_tensors...),
            hcat(node_js_tensors...), transpose(hcat(x0_tensors...)))
end


struct ModelInput
    likapprox_laparam::PyCall.PyObject
    likapprox_efflen::PyCall.PyObject
    likapprox_invhsb_params::Vector{PyCall.PyObject}
    likapprox_parent_idxs::Array
    likapprox_js::Array
    x0::Array{Float32, 2}
    sample_factors::Vector{Vector{String}}
    sample_names::Vector{String}
    feature::Symbol
    ts::Transcripts
    ts_metadata::TranscriptsMetadata
    output_filename::Nullable{String}
    output_format::Symbol
    gene_db::SQLite.DB
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


function write_effects(output_filename, factoridx, W, sigma, feature)
    println("Writing regression results to ", output_filename)

    db = SQLite.DB(output_filename)

    SQLite.execute!(db, "drop table if exists effects")
    SQLite.execute!(db,
        """
        create table effects
        ($(feature)_num INT, factor TEXT, w REAL, sigma REAL)
        """)

    ins_stmt = SQLite.Stmt(db, "insert into effects values (?1, ?2, ?3, ?4)")

    SQLite.execute!(db, "begin transaction")
    n = size(W, 2)
    for factor = sort(collect(keys(factoridx)))
        idx = factoridx[factor]
        for j in 1:n
            SQLite.bind!(ins_stmt, 1, j)
            SQLite.bind!(ins_stmt, 2, factor)
            SQLite.bind!(ins_stmt, 3, W[idx, j])
            SQLite.bind!(ins_stmt, 4, sigma[j])
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

