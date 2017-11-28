


function estimate_expression(input::ModelInput)
    if input.feature == :transcript
        return estimate_transcript_expression(input)
    elseif input.feature == :gene
        return estimate_gene_expression(input)
    elseif input.feature == :splicing
        return estimate_splicing_expression(input)
    else
        error("Expression estimates for $(feature)s not supported")
    end
end


function estimate_transcript_expression(input::ModelInput)
    num_samples, n = size(input.x0)
    x0_log = tf.log(tf.constant(input.x0))
    @show (num_samples, n)

    x, x_mu_param, x_sigma_param, x_mu, likapprox_laparam =
        transcript_quantification_model(input)

    println("Estimating...")

    qx_mu_param = tf.Variable(x0_log)
    qx_sigma_param = tf.nn[:softplus](tf.Variable(tf.fill([num_samples, n], -1.0f0)))
    qx = edmodels.MultivariateNormalDiag(qx_mu_param, qx_sigma_param)

    # qx_mu_mu_param = tf.Variable(tf.log(x0[1]))
    qx_mu_mu_param = tf.Variable(tf.reduce_mean(x0_log, 0))
    qx_mu_sigma_param = tf.nn[:softplus](tf.Variable(tf.fill([n], -1.0f0)))
    qx_mu = edmodels.MultivariateNormalDiag(qx_mu_mu_param, qx_mu_sigma_param)

    inference = ed.KLqp(PyDict(Dict(x => qx, x_mu => qx_mu)),
                        data=PyDict(Dict(likapprox_laparam => input.likapprox_laparam)))

    optimizer = tf.train[:AdamOptimizer](1e-2)
    inference[:run](n_iter=500, optimizer=optimizer)


    sess = ed.get_session()
    qx_mu_value    = sess[:run](qx_mu_param)
    qx_sigma_value = sess[:run](qx_sigma_param)

    est = sess[:run](tf.nn[:softmax](qx_mu_param, dim=-1))
    efflens = sess[:run](input.likapprox_efflen)

    # reset session and graph to free up memory in case more needs to be done
    tf.reset_default_graph()
    old_sess = ed.get_session()
    old_sess[:close]()
    ed.util[:graphs][:_ED_SESSION] = tf.InteractiveSession()

    # @show minimum(qx_mu_value), median(qx_mu_value), maximum(qx_mu_value)
    # @show minimum(qx_sigma_value), median(qx_sigma_value), maximum(qx_sigma_value)

    # TODO: this should be a temporary measure until we decide exactly how
    # results should be reported. Probably in sqlite or something.
    write_estimates("estimates.csv", input.sample_names, est)

    open("efflen.csv", "w") do out
        println(out, "transcript_num,efflen")
        for (i, efflen) in enumerate(efflens)
            println(out, i, ",", efflen)
        end
    end

    return qx_mu_value, qx_sigma_value
end


function estimate_gene_expression(input::ModelInput)
    num_samples, n = size(input.x0)
    num_features, gene_idxs, transcript_idxs, gene_names = gene_feature_matrix(input.ts, input.ts_metadata)
    num_aux_features = regularize_disjoint_feature_matrix!(gene_idxs, transcript_idxs, n)

    est = estimate_disjoint_feature_expression(input, gene_idxs, transcript_idxs,
                                               num_features + num_aux_features)

    if input.output_format == :csv
        output_filename = isnull(input.output_filename) ? "gene-expression.csv" : get(input.output_filename)
        write_gene_expression_csv(output_filename, input.sample_names, gene_names,
                                  num_aux_features, est)
    elseif input.output_format == :sqlite3
        error("Sqlite3 output for gene expression is not implemented.")
    end
end


function write_gene_expression_csv(output_filename, sample_names, gene_names,
                                   num_aux_features, est)
    n = size(est, 2) - num_aux_features
    @assert length(gene_names) == n
    open(output_filename, "w") do output
        println(output, "sample_name,gene_name,tpm")
        for (i, sample_name) in enumerate(sample_names)
            for (j, gene_name) in enumerate(gene_names)
                @printf(output, "%s,%s,%e\n", sample_name, gene_name, 1e6 * est[i,j])
            end
        end
    end
end


function estimate_splicing_expression(input::ModelInput)
    cassette_exons = get_cassette_exons(input.ts)
    # TODO: also include alt acceptor/donor sites and maybe alt UTRs

    # TODO: move this to another function
    db = input.gene_db
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
    ins_stmt = SQLite.Stmt(db,
        "insert into splicing_features values (?1, ?2, ?3, ?4, ?5, ?6, ?7)")
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
    end
    SQLite.execute!(db, "end transaction")

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
        "insert into splicing_feature_including_transcripts values (?1, ?2)")
    SQLite.execute!(db, "begin transaction")
    for (i, (intron, flanks)) in  enumerate(cassette_exons)
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

    # build feature matrix
    I = Int[]
    J = Int[]
    for (i, (intron, flanks)) in  enumerate(cassette_exons)
        @assert !isempty(intron.metadata)
        @assert !isempty(flanks.metadata[3])

        for id in flanks.metadata[3]
            push!(I, 2*i-1)
            push!(J, id)
        end

        for id in intron.metadata
            push!(I, 2*i)
            push!(J, id)
        end
    end
    m = 2*length(cassette_exons)

    F = tf.SparseTensor(indices=cat(2, I-1, J-1), values=tf.ones(length(I)),
                        dense_shape=[m, length(input.ts)])

    qy_mu_value, qy_sigma_value =
        estimate_feature_expression(input.likapprox_data, input.y0, input.sample_factors, F)
end


# function estimate_splicing_log_ratio(input::ModelInput)
#     cassette_exons = get_cassette_exons(input.ts)
#     I = Int[]
#     J = Int[]
#     V = Float32[]
#     for (i, (intron, flanks)) in  enumerate(cassette_exons)
#         for id in intron.metadata
#             push!(I, i)
#             push!(J, id)
#             push!(V, 1.0f0)
#         end

#         for id in flanks.metadata[3]
#             push!(I, i)
#             push!(J, id)
#             push!(V, -1.0f0)
#         end
#     end
#     m = length(cassette_exons)

#     F = tf.SparseTensor(indices=cat(2, I-1, J-1), values=V,
#                         dense_shape=[m, length(input.ts)])

#     qy_mu_value, qy_sigma_value =
#         estimate_feature_expression(input.likapprox_data, input.y0, input.sample_factors, F)
# end

function estimate_splicing_log_ratio(input::ModelInput)
    qy_mu_value, qy_sigma_value = estimate_splicing_expression(input)
    num_samples, n = size(qy_mu_value)

    inc_indexes = [i for i in 1:2:n]
    exc_indexes = [i for i in 2:2:n]

    qy_mu_inc_value = qy_mu_value[:,inc_indexes]
    qy_mu_exc_value = qy_mu_value[:,exc_indexes]

    qy_sigma_inc_value = qy_sigma_value[:,inc_indexes]
    qy_sigma_exc_value = qy_sigma_value[:,exc_indexes]

    qy_mu_ratio_value    = qy_mu_inc_value .- qy_mu_exc_value
    qy_sigma_ratio_value = qy_sigma_inc_value +- qy_sigma_exc_value

    @show minimum(qy_mu_inc_value), median(qy_mu_inc_value), maximum(qy_mu_inc_value)

    cassette_exons = get_cassette_exons(input.ts)

    println("EXTREME INC VALUES")
    for i in 1:length(qy_mu_inc_value)
        if qy_mu_inc_value[i] < -500.0
            @show (i, qy_mu_inc_value[i], qy_mu_exc_value[i], qy_sigma_inc_value[i], qy_sigma_exc_value[i])
            @show (length(cassette_exons[i][1].metadata), length(cassette_exons[i][2].metadata[3]))
        end
    end

    println("EXTREME EXC VALUES")
    for i in 1:length(qy_mu_exc_value)
        if qy_mu_exc_value[i] < -500.0
            @show (i, qy_mu_inc_value[i], qy_mu_exc_value[i], qy_sigma_inc_value[i], qy_sigma_exc_value[i])
            @show (length(cassette_exons[i][1].metadata), length(cassette_exons[i][2].metadata[3]))
        end
    end

    # tmp = sort(qy_mu_inc_value[2,:])
    # @show tmp[1:10]
    # @show tmp[end-10:end]

    exit()

    # @show minimum(qy_mu_exc_value), median(qy_mu_exc_value), maximum(qy_mu_exc_value)
    # @show minimum(qy_mu_ratio_value), median(qy_mu_ratio_value), maximum(qy_mu_ratio_value)
    # @show minimum(qy_sigma_ratio_value), median(qy_sigma_ratio_value), maximum(qy_sigma_ratio_value)

    return qy_mu_ratio_value, qy_sigma_ratio_value
end


"""
Build basic transcript quantification model with some shrinkage towards a pooled mean.
"""
function transcript_quantification_model(input::ModelInput)
    num_samples, n = size(input.x0)

    # y_mu: pooled mean
    # x_mu_mu0 = tf.constant(log(1/n), shape=[n])
    # x_mu_sigma0 = tf.constant(10.0, shape=[n])
    # TODO: 
    x_mu_mu0 = tf.constant(log(0.01 * 1/n), shape=[n])
    x_mu_sigma0 = tf.constant(5.0, shape=[n])
    x_mu = edmodels.MultivariateNormalDiag(x_mu_mu0, x_mu_sigma0)

    # x_sigma: variance around pooled mean
    x_sigma_mu0 = tf.constant(0.0, shape=[n])
    x_sigma_sigma0 = tf.constant(1.0, shape=[n])
    x_log_sigma = edmodels.MultivariateNormalDiag(x_sigma_mu0, x_sigma_sigma0)
    x_sigma = tf.exp(x_log_sigma)

    # y: quantification
    x_mu_param = tf.matmul(tf.ones([num_samples, 1]),
                           tf.expand_dims(x_mu, 0))

    x_sigma_param = tf.matmul(tf.ones([num_samples, 1]),
                              tf.expand_dims(x_sigma, 0))

    x = edmodels.MultivariateNormalDiag(x_mu_param, x_sigma_param)

    likapprox_laparam = rnaseq_approx_likelihood.RNASeqApproxLikelihood(
                    x=x,
                    efflens=input.likapprox_efflen,
                    invhsb_params=input.likapprox_invhsb_params,
                    value=input.likapprox_laparam)

    return x, x_mu_param, x_sigma_param, x_mu, likapprox_laparam
end


function estimate_disjoint_feature_expression(input::ModelInput, feature_idxs,
                                              transcript_idxs, num_features)
    num_samples, n = size(input.x0)

    # feature expression
    x_feature_mu_mu0 = tf.constant(log(0.01 * 1/num_features), shape=[num_features])
    x_feature_mu_sigma0 = tf.constant(5.0, shape=[num_features])
    x_feature_mu = edmodels.MultivariateNormalDiag(x_feature_mu_mu0,
                                                   x_feature_mu_sigma0)

    x_feature_sigma_mu0 = tf.constant(0.0, shape=[num_features])
    x_feature_sigma_sigma0 = tf.constant(1.0, shape=[num_features])
    x_feature_log_sigma = edmodels.MultivariateNormalDiag(x_feature_sigma_mu0,
                                                          x_feature_sigma_sigma0)
    x_feature_sigma = tf.exp(x_feature_log_sigma)

    x_feature_mu_param = tf.matmul(tf.ones([num_samples, 1]),
                                   tf.expand_dims(x_feature_mu, 0))

    x_feature_sigma_param = tf.matmul(tf.ones([num_samples, 1]),
                                      tf.expand_dims(x_feature_sigma, 0))

    x_feature = edmodels.MultivariateNormalDiag(x_feature_mu_param,
                                                x_feature_sigma_param)

    # within feature relative expression of feature constituents
    x_constituent_mu_mu0 = tf.constant(0.0, shape=[n])
    x_constituent_mu_sigma0 = tf.constant(5.0, shape=[n])
    x_constituent_mu = edmodels.MultivariateNormalDiag(x_constituent_mu_mu0,
                                                       x_constituent_mu_sigma0)

    x_constituent_sigma_mu0 = tf.constant(0.0, shape=[n])
    x_constituent_sigma_sigma0 = tf.constant(1.0, shape=[n])
    x_constituent_log_sigma = edmodels.MultivariateNormalDiag(x_constituent_sigma_mu0,
                                                              x_constituent_sigma_sigma0)
    x_constituent_sigma = tf.exp(x_constituent_log_sigma)

    x_constituent_mu_param = tf.matmul(tf.ones([num_samples, 1]),
                                       tf.expand_dims(x_constituent_mu, 0))

    x_constituent_sigma_param = tf.matmul(tf.ones([num_samples, 1]),
                                          tf.expand_dims(x_constituent_sigma, 0))

    x_constituent = edmodels.MultivariateNormalDiag(x_constituent_mu_param,
                                                    x_constituent_sigma_param)

    p = sortperm(feature_idxs)
    permute!(feature_idxs, p)
    permute!(transcript_idxs, p)
    x_constituent_indices = Array{Int32}((n, 2))
    for (k, (i, j)) in enumerate(zip(feature_idxs, transcript_idxs))
        x_constituent_indices[k, 1] = i-1
        x_constituent_indices[k, 2] = j-1
    end

    xs = []
    for (x_feature_i, x_constituent_i) in zip(tf.unstack(x_feature), tf.unstack(x_constituent))
        x_constituent_matrix = tf.SparseTensor(indices=x_constituent_indices,
                                               values=x_constituent_i,
                                               dense_shape=[num_features, n])
        x_constituent_matrix_softmax = tf.sparse_softmax(x_constituent_matrix)

        x_i = tf.log(tf.sparse_tensor_dense_matmul(x_constituent_matrix_softmax,
                                                   tf.expand_dims(tf.exp(x_feature_i), -1),
                                                   adjoint_a=true))

        push!(xs, x_i)
    end
    x = tf.squeeze(tf.stack(xs), axis=-1)

    likapprox_laparam = rnaseq_approx_likelihood.RNASeqApproxLikelihood(
                    x=x,
                    efflens=input.likapprox_efflen,
                    invhsb_params=input.likapprox_invhsb_params,
                    value=input.likapprox_laparam)

    # Inference
    # ---------

    # figure out some reasonable initial values
    feature_mu_initial     = zeros(Float32, (num_samples, num_features))
    constituent_mu_initial = zeros(Float32, (num_samples, n))
    for i in 1:num_samples
        for (j, k) in zip(feature_idxs, transcript_idxs)
            feature_mu_initial[i, j] += input.x0[i, k]
            constituent_mu_initial[i, k] = input.x0[i, k]
        end

        for (j, k) in zip(feature_idxs, transcript_idxs)
            constituent_mu_initial[i, k] /= feature_mu_initial[i, j]
        end
    end
    map!(log, feature_mu_initial, feature_mu_initial)
    map!(log, constituent_mu_initial, constituent_mu_initial)
    feature_mu_initial_mean = reshape(mean(feature_mu_initial, 1), (num_features,))
    constituent_mu_initial_mean = reshape(mean(constituent_mu_initial, 1), (n,))

    # qx_feature_mu_mu_param = tf.Variable(tf.fill([num_features], 0.0))
    qx_feature_mu_mu_param = tf.Variable(feature_mu_initial_mean)
    qx_feature_mu_sigma_param = tf.nn[:softplus](tf.Variable(tf.fill([num_features], -1.0f0)))
    qx_feature_mu = edmodels.MultivariateNormalDiag(qx_feature_mu_mu_param, qx_feature_mu_sigma_param)

    # qx_feature_mu_param = tf.Variable(tf.fill([num_samples, num_features], log(1/num_features)))
    qx_feature_mu_param = tf.Variable(feature_mu_initial)
    qx_feature_sigma_param = tf.nn[:softplus](tf.Variable(tf.fill([num_samples, num_features], -1.0f0)))
    qx_feature = edmodels.MultivariateNormalDiag(qx_feature_mu_param, qx_feature_sigma_param)

    # qx_constituent_mu_mu_param = tf.Variable(tf.fill([n], 0.0))
    qx_constituent_mu_mu_param = tf.Variable(constituent_mu_initial_mean)
    qx_constituent_mu_sigma_param = tf.nn[:softplus](tf.Variable(tf.fill([n], -1.0f0)))
    qx_constituent_mu = edmodels.MultivariateNormalDiag(qx_constituent_mu_mu_param, qx_constituent_mu_sigma_param)

    # qx_constituent_mu_param = tf.Variable(tf.fill([num_samples, n], 0.0))
    qx_constituent_mu_param = tf.Variable(constituent_mu_initial)
    qx_constituent_sigma_param = tf.nn[:softplus](tf.Variable(tf.fill([num_samples, n], -1.0f0)))
    qx_constituent = edmodels.MultivariateNormalDiag(qx_constituent_mu_param, qx_constituent_sigma_param)

    inference = ed.KLqp(PyDict(Dict(x_feature => qx_feature, x_feature_mu => qx_feature_mu,
                                    x_constituent => qx_constituent, x_constituent_mu => qx_constituent_mu)),
                        data=PyDict(Dict(likapprox_laparam => input.likapprox_laparam)))

    optimizer = tf.train[:AdamOptimizer](1e-2)
    inference[:run](n_iter=500, optimizer=optimizer)

    # output some estimates
    sess = ed.get_session()
    qx_mu_value    = sess[:run](qx_feature_mu_param)
    qx_sigma_value = sess[:run](qx_feature_sigma_param)

    est = sess[:run](tf.nn[:softmax](qx_feature_mu_param, dim=-1))
    efflens = sess[:run](input.likapprox_efflen)

    # reset session and graph to free up memory in case more needs to be done
    tf.reset_default_graph()
    old_sess = ed.get_session()
    old_sess[:close]()
    ed.util[:graphs][:_ED_SESSION] = tf.InteractiveSession()

    return est
end

EXTRUDER_MODELS["expression"] = estimate_expression
EXTRUDER_MODELS["splicing"]   = estimate_splicing_log_ratio

