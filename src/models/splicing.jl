
# Models explicitly considering splicing rates. Splicing a particularly tricky
# issue due to the weirdness of the transcript expression -> splicing
# transformation. We try to overcome this by building an approximation to the
# splicing rate likelihood function and basing all our models on that.

"""
Build tensorflow transformation from expression vectors to splicing log-ratios.
"""
function transcript_expression_to_splicing_log_ratios(
                num_features, n,
                feature_idxs, feature_transcript_idxs,
                antifeature_idxs, antifeature_transcript_idxs, x)

    feature_matrix = tf.SparseTensor(
        indices=hcat(feature_idxs .- 1, feature_transcript_idxs .- 1),
        values=tf.ones(length(feature_idxs)),
        dense_shape=[num_features, n])

    antifeature_matrix = tf.SparseTensor(
        indices=hcat(antifeature_idxs .- 1, antifeature_transcript_idxs .- 1),
        values=tf.ones(length(antifeature_idxs)),
        dense_shape=[num_features, n])

    splice_lrs = Any[]
    for x_i in tf.unstack(x)
        x_i_ = tf.expand_dims(x_i, -1)
        # x_i = tf_print_span(x_i, "x_i")

        feature_expr     = tf.sparse_tensor_dense_matmul(feature_matrix, x_i_)
        antifeature_expr = tf.sparse_tensor_dense_matmul(antifeature_matrix, x_i_)

        # feature_expr = tf_print_span(feature_expr, "feature_expr")
        # antifeature_expr = tf_print_span(antifeature_expr, "antifeature_expr")

        push!(splice_lrs, tf.log(feature_expr) - tf.log(antifeature_expr))
    end

    splice_lr = tf.squeeze(tf.stack(splice_lrs), axis=-1)
    # splice_lr = tf_print_span(splice_lr, "splice_lr span")
    return splice_lr
end


"""
There isn't a tractable way to evaluate P(reads|splicing ratios), however we
can efficiently sample in proportion to it. We use that fact to build a
Logit-Normal approximation to the splicing likelihood by minimizing KL(p,q)
(not KL(q,p) as is more common in variational inference). This is pretty
simple to do.

    argmin KL(p,q) = argmin -E[log q(x)] + E[log p(x)]

        with expectations wrt to p

    = argmax E[log q(x)]

So all we have to do is optimize q's parameters using stochastic gradient descent
against samples drawn from p.
"""
function approximate_splicing_likelihood(input::ModelInput)
    num_samples, n = size(input.loaded_samples.x0_values)

    (num_features,
     feature_idxs, feature_transcript_idxs,
     antifeature_idxs, antifeature_transcript_idxs) = splicing_features(input)

    # Approximation

    x_feature_loc = polee_py.ImproperPrior(
        value=tf.zeros([num_samples, num_features]))

    x_feature_scale = polee_py.ImproperPrior(
        value=tf.zeros([num_samples, num_features]))

    x_feature = edmodels.NormalWithSoftplusScale(
        loc=x_feature_loc, scale=x_feature_scale, name="x_feature")

    # Inference
    qx_feature_loc_param = tf.Variable(
        tf.zeros([num_samples, num_features]), name="qx_feature_loc_param")
    # qx_feature_loc_param = tf_print_span(qx_feature_loc_param, "qx_feature_loc_param")
    qx_feature_loc = edmodels.PointMass(qx_feature_loc_param)

    qx_feature_scale_param = tf.Variable(
        tf.zeros([num_samples, num_features]), name="qx_feature_scale_param")
    # qx_feature_scale_param = tf_print_span(qx_feature_scale_param, "qx_feature_scale_param")
    qx_feature_scale = edmodels.PointMass(qx_feature_scale_param)

    T = x -> transcript_expression_to_splicing_log_ratios(
                num_features, n, feature_idxs, feature_transcript_idxs,
                antifeature_idxs, antifeature_transcript_idxs, x)

    optimizer = tf.train[:AdamOptimizer](1e-2)
    latent_vars = Dict(
        x_feature_loc    => qx_feature_loc,
        x_feature_scale  => qx_feature_scale
    )
    run_implicit_model_map_inference(input, x_feature, T, latent_vars, 500, optimizer)

    sess = ed.get_session()
    return (sess[:run](qx_feature_loc), sess[:run](qx_feature_scale))
end


"""
Identify alternative splicing features (e.g. cassette exons), add to the gene
database, and return.

Returns:
    num_features: Number of alternative splicing features.
    feature_idxs and feature_transcript_idxs: together a mapping between
        features and transcripts which include that feature
    antifeature_idxs and antifeature_transcript_idxs: together a mapping between
        features and transcripts which exclude that feature
"""
function splicing_features(input::ModelInput)
    println("")
    cassette_exons = get_cassette_exons(input.ts)
    alt_donacc_sites, retained_introns = get_alt_donor_acceptor_sites(input.ts)

    println("Read ", length(cassette_exons), " cassette exons")
    println("     ", length(alt_donacc_sites), " alternate acceptor/donor sites")
    println("     ", length(retained_introns), " retained introns")
    # TODO: alt 5'-most exons
    # TODO: alt 3'-most exons

    @time write_splicing_features_to_gene_db(
        input.gene_db, cassette_exons, alt_donacc_sites, retained_introns)

    feature_idxs = Int32[]
    feature_transcript_idxs = Int32[]
    antifeature_idxs = Int32[]
    antifeature_transcript_idxs = Int32[]
    num_features =
        length(cassette_exons) + length(alt_donacc_sites) + length(retained_introns)

    # cassette exons
    for (i, (intron, flanks)) in  enumerate(cassette_exons)
        @assert !isempty(intron.metadata)
        @assert !isempty(flanks.metadata[3])

        for id in flanks.metadata[3]
            push!(feature_idxs, i)
            push!(feature_transcript_idxs, id)
        end

        for id in intron.metadata
            push!(antifeature_idxs, i)
            push!(antifeature_transcript_idxs, id)
        end
    end

    # alt donor/acceptor sites
    for (i, alt_donacc_site) in enumerate(alt_donacc_sites)
        idx = i + length(cassette_exons)

        for id in alt_donacc_site.metadata[3]
            push!(feature_idxs, idx)
            push!(feature_transcript_idxs, id)
        end

        for id in alt_donacc_site.metadata[4]
            push!(antifeature_idxs, idx)
            push!(antifeature_transcript_idxs, id)
        end
    end

    # retained introns
    for (i, retained_intron) in enumerate(retained_introns)
        idx = i + length(cassette_exons) + length(alt_donacc_sites)

        for id in retained_intron.metadata[1]
            push!(feature_idxs, idx)
            push!(feature_transcript_idxs, id)
        end

        for id in retained_intron.metadata[2]
            push!(antifeature_idxs, idx)
            push!(antifeature_transcript_idxs, id)
        end
    end

    return num_features,
           feature_idxs, feature_transcript_idxs,
           antifeature_idxs, antifeature_transcript_idxs
end


function write_splicing_features_to_gene_db(
    db, cassette_exons, alt_donacc_sites, retained_introns)

    SQLite.execute!(db, "drop table if exists splicing_features")
    SQLite.execute!(db,
        """
        create table splicing_features
        (
            feature_num INT PRIMARY KEY,
            type TEXT,
            seqname TEXT,
            included_first INT,
            included_last INT,
            excluded_first INT,
            excluded_last INT
        )
        """)

    SQLite.execute!(db, "drop table if exists splicing_feature_including_transcripts")
    SQLite.execute!(db,
        """
        create table splicing_feature_including_transcripts
        (
            feature_num INT KEY,
            transcript_num INT
        )
        """)

    SQLite.execute!(db, "drop table if exists splicing_feature_excluding_transcripts")
    SQLite.execute!(db,
        """
        create table splicing_feature_excluding_transcripts
        (
            feature_num INT KEY,
            transcript_num INT
        )
        """)

    inc_ins_stmt = SQLite.Stmt(db,
        "insert into splicing_feature_including_transcripts values (?1, ?2)")
    exc_ins_stmt = SQLite.Stmt(db,
        "insert into splicing_feature_excluding_transcripts values (?1, ?2)")
    ins_stmt = SQLite.Stmt(db,
        "insert into splicing_features values (?1, ?2, ?3, ?4, ?5, ?6, ?7)")

    # cassette exons
    SQLite.execute!(db, "begin transaction")
    for (i, (intron, flanks)) in  enumerate(cassette_exons)
        SQLite.bind!(ins_stmt, 1, i)
        SQLite.bind!(ins_stmt, 2, "cassette_exon")
        SQLite.bind!(ins_stmt, 3, intron.seqname)
        SQLite.bind!(ins_stmt, 4, flanks.metadata[1])
        SQLite.bind!(ins_stmt, 5, flanks.metadata[2])
        SQLite.bind!(ins_stmt, 6, flanks.first)
        SQLite.bind!(ins_stmt, 7, flanks.last)
        SQLite.execute!(ins_stmt)

        SQLite.bind!(inc_ins_stmt, 1, i)
        SQLite.bind!(exc_ins_stmt, 1, i)

        for id in flanks.metadata[3]
            SQLite.bind!(inc_ins_stmt, 2, id)
            SQLite.execute!(inc_ins_stmt)
        end

        for id in intron.metadata
            SQLite.bind!(exc_ins_stmt, 2, id)
            SQLite.execute!(exc_ins_stmt)
        end
    end
    SQLite.execute!(db, "end transaction")

    # alt donor/acceptor sites
    SQLite.execute!(db, "begin transaction")
    for (i, site) in enumerate(alt_donacc_sites)
        idx = i + length(cassette_exons)

        typ = site.first == site.metadata[1] ?
            site.strand == STRAND_POS ? "alt acceptor site" : "alt donor site" :
            site.strand == STRAND_POS ? "alt donor site": "alt acceptor site"

        SQLite.bind!(ins_stmt, 1, idx)
        SQLite.bind!(ins_stmt, 2, typ)
        SQLite.bind!(ins_stmt, 3, site.seqname)
        SQLite.bind!(ins_stmt, 4, site.first)
        SQLite.bind!(ins_stmt, 5, site.last)
        SQLite.bind!(ins_stmt, 6, site.metadata[1])
        SQLite.bind!(ins_stmt, 7, site.metadata[2])
        SQLite.execute!(ins_stmt)

        SQLite.bind!(inc_ins_stmt, 1, idx)
        SQLite.bind!(exc_ins_stmt, 1, idx)

        for id in site.metadata[3]
            SQLite.bind!(inc_ins_stmt, 2, id)
            SQLite.execute!(inc_ins_stmt)
        end

        for id in site.metadata[4]
            SQLite.bind!(exc_ins_stmt, 2, id)
            SQLite.execute!(exc_ins_stmt)
        end
    end
    SQLite.execute!(db, "end transaction")

    # retained intron
    SQLite.execute!(db, "begin transaction")
    for (i, retained_intron) in enumerate(retained_introns)
        idx = i + length(cassette_exons) + length(alt_donacc_sites)

        SQLite.bind!(ins_stmt, 1, idx)
        SQLite.bind!(ins_stmt, 2, "retained intron")
        SQLite.bind!(ins_stmt, 3, retained_intron.seqname)
        SQLite.bind!(ins_stmt, 4, retained_intron.first)
        SQLite.bind!(ins_stmt, 5, retained_intron.last)
        SQLite.bind!(ins_stmt, 6, -1)
        SQLite.bind!(ins_stmt, 7, -1)
        SQLite.execute!(ins_stmt)

        SQLite.bind!(inc_ins_stmt, 1, idx)
        SQLite.bind!(exc_ins_stmt, 1, idx)

        for id in retained_intron.metadata[1]
            SQLite.bind!(inc_ins_stmt, 2, id)
            SQLite.execute!(inc_ins_stmt)
        end

        for id in retained_intron.metadata[2]
            SQLite.bind!(exc_ins_stmt, 2, id)
            SQLite.execute!(exc_ins_stmt)
        end
    end
    SQLite.execute!(db, "end transaction")
end

