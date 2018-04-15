#!/usr/bin/env julia

# This program written to analyze a specific experiment involving comparisons
# across species. Is to place samples in the same latent space representing
# expression of orthologous groups of transcripts. Orthologs were identified
# using OrthoMCL.

import Polee
using PyCall
using YAML

@pyimport tensorflow as tf
@pyimport edward as ed
@pyimport edward.models as edmodels

function main()
    ortholog_groups_filename = ARGS[1]

    species_loaded_samples       = Polee.LoadedSamples[]
    species_transcripts          = Polee.Transcripts[]
    species_transcripts_metadatas = Polee.TranscriptsMetadata[]

    # Load samples and transcripts for every species
    num_samples = 0
    i = 2
    while i <= length(ARGS)
        if i + 1 > length(ARGS)
            error("Last gff filename needs to be matches with an experiment")
        end
        genes_filename = ARGS[i]
        experiment_filename = ARGS[i+1]

        ts, ts_metadata = Polee.Transcripts(genes_filename)

        spec = YAML.load_file(experiment_filename)
        loaded_samples = Polee.load_samples_from_specification(
            spec, ts, ts_metadata)

        push!(species_loaded_samples, loaded_samples)
        push!(species_transcripts, ts)
        push!(species_transcripts_metadatas, ts_metadata)

        num_samples += length(loaded_samples.sample_names)

        i += 2
    end
    num_taxons = length(species_loaded_samples)

    # Load orthoMCL ortholog groups
    ortholog_groups = Set{String}[]
    ortholog_group_by_id = Dict{String, Int}()
    taxons = Set{String}()
    for line in eachline(ortholog_groups_filename)
        group_name, transcripts_str = split(line, '\t')
        transcripts = split(transcripts_str, ',')

        # remove taxon indicators from names, exclude groups that don't
        # have transcripts from all species.
        empty!(taxons)
        for i in 1:length(transcripts)
            mat = match(r"^([^\|]*)\|(.*)$", transcripts[i])
            taxon = mat.captures[1]
            id = mat.captures[2]
            transcripts[i] = id
            push!(taxons, taxon)
        end

        if length(taxons) != num_taxons
            continue
        end

        group_idx = 1 + length(ortholog_groups)
        push!(ortholog_groups, Set(transcripts))
        for transcript_ in transcripts
            transcript = String(transcript_)
            @assert !haskey(ortholog_group_by_id, transcript)
            ortholog_group_by_id[transcript] = group_idx
        end
    end
    num_orth_groups = length(ortholog_groups)
    println("Read ", num_orth_groups, " ortholog groups")

    latent_vars = Dict()
    data = Dict()

    # TODO: I think this really should be a hierarchical model. Since it's on a
    # floating scale, it may not be super meaningful otherwise.
    x_og_mu0 = log(1.0f0/num_orth_groups)
    x_og_sigma0 = 5.0
    x_og = edmodels.Normal(
        loc=tf.fill([num_samples, num_orth_groups], x_og_mu0),
        scale=x_og_sigma0)

    idxoff = 0
    for taxon_idx in 1:num_taxons
        ts = species_transcripts[taxon_idx]
        ts_metadata = species_transcripts_metadatas[taxon_idx]

        species_n = length(ts)
        species_num_samples = length(species_loaded_samples[taxon_idx].sample_names)

        # slice out ortholog group expression for this species
        # ----------------------------------------------------

        x_ortho = tf.slice(x_og, Int32[idxoff, 0], Int32[species_num_samples, num_orth_groups])
        idxoff += species_num_samples

        # within group mixtures
        # ---------------------

        nongrouped_mask_values = zeros(Float32, species_n)
        I = Int[]
        J = Int[]
        for t in ts
            if haskey(ortholog_group_by_id, t.metadata.name)
                push!(I, ortholog_group_by_id[t.metadata.name])
                push!(J, t.metadata.id)
            else
                nongrouped_mask_values[t.metadata.id] = 1.0f0
            end
        end
        nongrouped_mask = tf.constant(nongrouped_mask_values)
        @show nongrouped_mask
        grouped_n = length(I)
        p = sortperm(I)
        permute!(I, p)
        permute!(J, p)

        x_const_mat_idx = Array{Int32}((grouped_n, 2))
        for (k, (i, j)) in enumerate(zip(I, J))
            x_const_mat_idx[k, 1] = i - 1
            x_const_mat_idx[k, 2] = j - 1
        end

        x_constituent_mu_param = tf.fill([species_num_samples, grouped_n], 0.0f0)
        x_constituent_sigma_param = tf.fill([species_num_samples, grouped_n], 10.0f0)

        x_constituent = edmodels.Normal(loc=x_constituent_mu_param,
                                        scale=x_constituent_sigma_param)
        # x_constituent_ = Polee.tf_print_span(x_constituent, "x_constituent")

        xs = []
        for (x_ortho_i, x_const_i) in zip(tf.unstack(x_ortho), tf.unstack(x_constituent))
            x_const_mat = tf.SparseTensor(
                indices=x_const_mat_idx,
                values=x_const_i,
                dense_shape=[num_orth_groups, species_n])

            x_const_mat_softmax = tf.sparse_softmax(x_const_mat)

            x_exp_i = tf.sparse_tensor_dense_matmul(x_const_mat_softmax,
                                                    tf.expand_dims(tf.exp(x_ortho_i), -1),
                                                    adjoint_a=true)
            x_exp_i = tf.squeeze(x_exp_i, -1) + nongrouped_mask

            # x_exp_i = Polee.tf_print_span(x_exp_i, "x_exp_i")
            x_i = tf.log(x_exp_i)

            # x_i = tf.log(tf.sparse_tensor_dense_matmul(x_const_mat_softmax,
            #                                         tf.expand_dims(tf.exp(x_ortho_i), -1),
            #                                         adjoint_a=true))
            push!(xs, x_i)
        end
        x = tf.stack(xs)
        # x = Polee.tf_print_span(x, "x")

        # non-grouped transcript expression
        # ---------------------------------

        nongrouped_n = species_n - length(I)
        x_ng = edmodels.Normal(
            loc=tf.fill([species_num_samples, nongrouped_n], log(1.0f0/nongrouped_n)),
            scale=10.0f0)

        # build a sparse matrix to map nongrouped transcript indices to
        # transcript indices
        I = Int[]
        J = Int[]
        for t in ts
            if !haskey(ortholog_group_by_id, t.metadata.name)
                push!(I, t.metadata.id)
                push!(J, length(J) + 1)
            end
        end
        p = sortperm(I)
        permute!(I, p)
        permute!(J, p)

        x_ng_mat_idx = Array{Int32}((nongrouped_n, 2))
        for (k, (i, j)) in enumerate(zip(I, J))
            x_ng_mat_idx[k, 1] = i - 1
            x_ng_mat_idx[k, 2] = j - 1
        end

        x_ng_mat = tf.SparseTensor(
            indices=x_ng_mat_idx,
            values=tf.ones(length(I)),
            dense_shape=[species_n, nongrouped_n])

        x_ng_t = tf.transpose(tf.sparse_tensor_dense_matmul(x_ng_mat, tf.transpose(x_ng)))
        # x_ng_t = Polee.tf_print_span(x_ng_t, "x_ng_t")

        x += x_ng_t

        likapprox = Polee.RNASeqApproxLikelihood(species_loaded_samples[taxon_idx], x)
        data[likapprox] = Float32[]

        latent_vars[x_constituent] = edmodels.NormalWithSoftplusScale(
            loc=tf.Variable(tf.fill([species_num_samples, grouped_n], 0.0f0)),
            scale=tf.Variable(tf.fill([species_num_samples, grouped_n], 0.0f0)))

        latent_vars[x_ng] = edmodels.NormalWithSoftplusScale(
            loc=tf.Variable(tf.fill([species_num_samples, nongrouped_n], log(1.0f0/nongrouped_n))),
            scale=tf.Variable(tf.fill([species_num_samples, nongrouped_n], 0.0f0)))
    end

    # Inference
    # ---------

    latent_vars[x_og] = edmodels.NormalWithSoftplusScale(
        loc=tf.Variable(tf.fill([num_samples, num_orth_groups], x_og_mu0)),
        scale=tf.Variable(tf.fill([num_samples, num_orth_groups], 0.0f0)))

    inference= ed.KLqp(latent_vars, data)
    optimizer = tf.train[:AdamOptimizer](0.05)

    init_feed_dict = reduce(merge, [ls.init_feed_dict for ls in species_loaded_samples])
    Polee.run_inference(init_feed_dict, inference, 1000, optimizer)

    sess = ed.get_session()
    open("latent-ortholog-expression.csv", "w") do output
        println(output, "sample,orthogroup,expression")
        x_est = sess[:run](latent_vars[x_og][:mean]())
        i = 1
        for k in 1:length(species_loaded_samples)
            ls = species_loaded_samples[k]
            for name in ls.sample_names
                for j in 1:size(x_est, 2)
                    println(output, name, ",", j, ",", x_est[i, j])
                end
                i += 1
            end
        end
    end
end



main()
