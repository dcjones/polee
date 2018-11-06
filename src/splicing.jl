
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

    # TODO: make sure indices are in the right order
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
function approximate_splicing_likelihood(input::ModelInput, sess)
    num_samples, n = size(input.loaded_samples.x0_values)

    (num_features,
     feature_idxs, feature_transcript_idxs,
     antifeature_idxs, antifeature_transcript_idxs) = splicing_features(input)

    feature_indices = hcat(feature_idxs .- 1, feature_transcript_idxs .- 1)
    antifeature_indices = hcat(antifeature_idxs .- 1, antifeature_transcript_idxs .- 1)

    qx_feature_loc, qx_feature_scale = polee_py[:approximate_splicing_likelihood](
        input.loaded_samples.init_feed_dict, input.loaded_samples.variables,
        num_samples, num_features, n, feature_indices, antifeature_indices, sess)

    return (qx_feature_loc, qx_feature_scale)
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
    cassette_exons, mutex_exons = get_cassette_exons(input.ts)
    alt_donacc_sites, retained_introns = get_alt_donor_acceptor_sites(input.ts)
    alt_fp_ends, alt_tp_ends = get_alt_fp_tp_ends(input.ts, input.ts_metadata)

    println("Read ", length(cassette_exons), " cassette exons")
    println("     ", length(mutex_exons), " mutually exclusive exons")
    println("     ", length(alt_donacc_sites), " alternate acceptor/donor sites")
    println("     ", length(retained_introns), " retained introns")
    println("     ", length(alt_fp_ends), " alternate 5' ends")
    println("     ", length(alt_tp_ends), " alternate 3' ends")


    @time write_splicing_features_to_gene_db(
        input.gene_db, cassette_exons, mutex_exons,
        alt_donacc_sites, retained_introns,
        alt_fp_ends, alt_tp_ends)

    # exit()

    feature_idxs = Int32[]
    feature_transcript_idxs = Int32[]
    antifeature_idxs = Int32[]
    antifeature_transcript_idxs = Int32[]
    num_features =
        length(cassette_exons) + length(mutex_exons) + length(alt_donacc_sites) +
        length(retained_introns) + length(alt_fp_ends) + length(alt_tp_ends)

    feature_id = 0

    # cassette exons
    for (intron, flanks) in cassette_exons
        feature_id += 1
        @assert !isempty(intron.metadata)
        @assert !isempty(flanks.metadata[3])

        for id in flanks.metadata[3]
            push!(feature_idxs, feature_id)
            push!(feature_transcript_idxs, id)
        end

        for id in intron.metadata
            push!(antifeature_idxs, feature_id)
            push!(antifeature_transcript_idxs, id)
        end
    end

    # mutex exons
    for (exon_a, exon_b) in mutex_exons
        feature_id += 1

        for id in exon_a.metadata
            push!(feature_idxs, feature_id)
            push!(feature_transcript_idxs, id)
        end

        for id in exon_b.metadata
            push!(antifeature_idxs, feature_id)
            push!(antifeature_transcript_idxs, id)
        end
    end

    # alt donor/acceptor sites
    for alt_donacc_site in alt_donacc_sites
        feature_id += 1

        for id in alt_donacc_site.metadata[3]
            push!(feature_idxs, feature_id)
            push!(feature_transcript_idxs, id)
        end

        for id in alt_donacc_site.metadata[4]
            push!(antifeature_idxs, feature_id)
            push!(antifeature_transcript_idxs, id)
        end
    end

    # retained introns
    for retained_intron in retained_introns
        feature_id += 1

        for id in retained_intron.metadata[1]
            push!(feature_idxs, feature_id)
            push!(feature_transcript_idxs, id)
        end

        for id in retained_intron.metadata[2]
            push!(antifeature_idxs, feature_id)
            push!(antifeature_transcript_idxs, id)
        end
    end

    # alt 5' ends
    for alt_fp_end in alt_fp_ends
        feature_id += 1

        for id in alt_fp_end.metadata[3]
            push!(feature_idxs, feature_id)
            push!(feature_transcript_idxs, id)
        end

        for id in alt_fp_end.metadata[4]
            push!(antifeature_idxs, feature_id)
            push!(antifeature_transcript_idxs, id)
        end
    end

    # alt 3' ends
    for alt_tp_end in alt_tp_ends
        feature_id += 1

        for id in alt_tp_end.metadata[3]
            push!(feature_idxs, feature_id)
            push!(feature_transcript_idxs, id)
        end

        for id in alt_tp_end.metadata[4]
            push!(antifeature_idxs, feature_id)
            push!(antifeature_transcript_idxs, id)
        end
    end

    return num_features,
           feature_idxs, feature_transcript_idxs,
           antifeature_idxs, antifeature_transcript_idxs
end


function write_splicing_features_to_gene_db(
    db, cassette_exons, mutex_exons, alt_donacc_sites, retained_introns,
    alt_fp_ends, alt_tp_ends)

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

    feature_id = 0

    # cassette exons
    SQLite.execute!(db, "begin transaction")
    for (intron, flanks) in  cassette_exons
        feature_id += 1
        SQLite.bind!(ins_stmt, 1, feature_id)
        SQLite.bind!(ins_stmt, 2, "cassette_exon")
        SQLite.bind!(ins_stmt, 3, intron.seqname)
        SQLite.bind!(ins_stmt, 4, flanks.metadata[1])
        SQLite.bind!(ins_stmt, 5, flanks.metadata[2])
        SQLite.bind!(ins_stmt, 6, flanks.first)
        SQLite.bind!(ins_stmt, 7, flanks.last)
        SQLite.execute!(ins_stmt)

        SQLite.bind!(inc_ins_stmt, 1, feature_id)
        SQLite.bind!(exc_ins_stmt, 1, feature_id)

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

    # mutex exons
    SQLite.execute!(db, "begin transaction")
    for (exon_a, exon_b) in mutex_exons
        feature_id += 1
        SQLite.bind!(ins_stmt, 1, feature_id)
        SQLite.bind!(ins_stmt, 2, "mutex_exon")
        SQLite.bind!(ins_stmt, 3, exon_a.seqname)
        SQLite.bind!(ins_stmt, 4, exon_a.first)
        SQLite.bind!(ins_stmt, 5, exon_a.last)
        SQLite.bind!(ins_stmt, 6, exon_b.first)
        SQLite.bind!(ins_stmt, 7, exon_b.last)
        SQLite.execute!(ins_stmt)

        SQLite.bind!(inc_ins_stmt, 1, feature_id)
        SQLite.bind!(exc_ins_stmt, 1, feature_id)

        for id in exon_a.metadata
            SQLite.bind!(inc_ins_stmt, 2, id)
            SQLite.execute!(inc_ins_stmt)
        end

        for id in exon_b.metadata
            SQLite.bind!(exc_ins_stmt, 2, id)
            SQLite.execute!(exc_ins_stmt)
        end
    end
    SQLite.execute!(db, "end transaction")

    # alt donor/acceptor sites
    SQLite.execute!(db, "begin transaction")
    for site in alt_donacc_sites
        feature_id += 1

        typ = site.first == site.metadata[1] ?
            site.strand == STRAND_POS ? "alt acceptor site" : "alt donor site" :
            site.strand == STRAND_POS ? "alt donor site" : "alt acceptor site"

        SQLite.bind!(ins_stmt, 1, feature_id)
        SQLite.bind!(ins_stmt, 2, typ)
        SQLite.bind!(ins_stmt, 3, site.seqname)
        SQLite.bind!(ins_stmt, 4, site.first)
        SQLite.bind!(ins_stmt, 5, site.last)
        SQLite.bind!(ins_stmt, 6, site.metadata[1])
        SQLite.bind!(ins_stmt, 7, site.metadata[2])
        SQLite.execute!(ins_stmt)

        SQLite.bind!(inc_ins_stmt, 1, feature_id)
        SQLite.bind!(exc_ins_stmt, 1, feature_id)

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
    for retained_intron in retained_introns
        feature_id += 1

        SQLite.bind!(ins_stmt, 1, feature_id)
        SQLite.bind!(ins_stmt, 2, "retained intron")
        SQLite.bind!(ins_stmt, 3, retained_intron.seqname)
        SQLite.bind!(ins_stmt, 4, retained_intron.first)
        SQLite.bind!(ins_stmt, 5, retained_intron.last)
        SQLite.bind!(ins_stmt, 6, -1)
        SQLite.bind!(ins_stmt, 7, -1)
        SQLite.execute!(ins_stmt)

        SQLite.bind!(inc_ins_stmt, 1, feature_id)
        SQLite.bind!(exc_ins_stmt, 1, feature_id)

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

    # alt 5' ends
    SQLite.execute!(db, "begin transaction")
    for alt_fp_end in alt_fp_ends
        feature_id += 1

        SQLite.bind!(ins_stmt, 1, feature_id)
        SQLite.bind!(ins_stmt, 2, "alt 5' end")
        SQLite.bind!(ins_stmt, 3, alt_fp_end.seqname)
        SQLite.bind!(ins_stmt, 4, alt_fp_end.first)
        SQLite.bind!(ins_stmt, 5, alt_fp_end.last)
        SQLite.bind!(ins_stmt, 6, -1)
        SQLite.bind!(ins_stmt, 7, -1)
        SQLite.execute!(ins_stmt)

        SQLite.bind!(inc_ins_stmt, 1, feature_id)
        SQLite.bind!(exc_ins_stmt, 1, feature_id)

        for id in alt_fp_end.metadata[3]
            SQLite.bind!(inc_ins_stmt, 2, id)
            SQLite.execute!(inc_ins_stmt)
        end

        for id in alt_fp_end.metadata[4]
            SQLite.bind!(exc_ins_stmt, 2, id)
            SQLite.execute!(exc_ins_stmt)
        end
    end
    SQLite.execute!(db, "end transaction")

    # alt 3' ends
    SQLite.execute!(db, "begin transaction")
    for alt_tp_end in alt_tp_ends
        feature_id += 1

        SQLite.bind!(ins_stmt, 1, feature_id)
        SQLite.bind!(ins_stmt, 2, "alt 5' end")
        SQLite.bind!(ins_stmt, 3, alt_tp_end.seqname)
        SQLite.bind!(ins_stmt, 4, alt_tp_end.first)
        SQLite.bind!(ins_stmt, 5, alt_tp_end.last)
        SQLite.bind!(ins_stmt, 6, -1)
        SQLite.bind!(ins_stmt, 7, -1)
        SQLite.execute!(ins_stmt)

        SQLite.bind!(inc_ins_stmt, 1, feature_id)
        SQLite.bind!(exc_ins_stmt, 1, feature_id)

        for id in alt_tp_end.metadata[3]
            SQLite.bind!(inc_ins_stmt, 2, id)
            SQLite.execute!(inc_ins_stmt)
        end

        for id in alt_tp_end.metadata[4]
            SQLite.bind!(exc_ins_stmt, 2, id)
            SQLite.execute!(exc_ins_stmt)
        end
    end
    SQLite.execute!(db, "end transaction")
end

