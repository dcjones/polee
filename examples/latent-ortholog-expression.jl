#!/usr/bin/env julia

# This program written to analyze a specific experiment involving comparisons
# across species. Is to place samples in the same latent space representing
# expression of orthologous groups of transcripts. Orthologs were identified
# using OrthoMCL.

import Polee
using PyCall
using YAML

@pyimport tensorflow as tf
@pyimport tensorflow.contrib.distributions as tfdist
@pyimport edward as ed
@pyimport edward.models as edmodels


const baseline_treatment = "2"
# const baseline_treatment = nothing

function main()
    # Which treatment to use as the baseline. 3 is serum starvation.

    model_name = ARGS[1]
    ortholog_groups_filename = ARGS[2]

    species_loaded_samples       = Polee.LoadedSamples[]
    species_transcripts          = Polee.Transcripts[]
    species_transcripts_metadatas = Polee.TranscriptsMetadata[]

    # Load samples and transcripts for every species
    num_samples = 0
    i = 3
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
    ortholog_group_names = String[]
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

        if length(taxons) < num_taxons
            continue
        end

        group_idx = 1 + length(ortholog_groups)
        push!(ortholog_group_names, group_name)
        push!(ortholog_groups, Set(transcripts))
        for transcript_ in transcripts
            transcript = String(transcript_)
            @assert !haskey(ortholog_group_by_id, transcript)
            ortholog_group_by_id[transcript] = group_idx
        end
    end
    num_orth_groups = length(ortholog_groups)
    println("Read ", num_orth_groups, " ortholog groups")

    data = Dict()

    if model_name == "pca"
        num_components = 2
        x_og, vars, latent_vars, nonbaseline_sample_mat = orthogroup_pca(
            species_loaded_samples, species_transcripts, ortholog_group_by_id,
            num_taxons, num_samples, num_orth_groups, num_components)
    elseif model_name == "regression"
        x_og, vars, latent_vars, species_factors, cond_factors = orthogroup_regression(
            species_loaded_samples, species_transcripts, ortholog_group_by_id,
            num_taxons, num_samples, num_orth_groups)
    else
        error("$(model_name) is not a valid model")
    end

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

            x_i = tf.log(x_exp_i)
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

        if model_name == "regression"
            latent_vars[x_constituent] = edmodels.NormalWithSoftplusScale(
                loc=tf.Variable(tf.fill([species_num_samples, grouped_n], 0.0f0), name="qx_constituent_loc"),
                scale=tf.Variable(tf.fill([species_num_samples, grouped_n], 0.0f0), name="qx_constituent_scale"))

            latent_vars[x_ng] = edmodels.NormalWithSoftplusScale(
                loc=tf.Variable(tf.fill([species_num_samples, nongrouped_n], log(1.0f0/nongrouped_n)), name="qx_ng_loc"),
                scale=tf.Variable(tf.fill([species_num_samples, nongrouped_n], 0.0f0), name="qx_ng_scale"))
        elseif model_name == "pca"
            latent_vars[x_constituent] = edmodels.PointMass(
                tf.Variable(tf.fill([species_num_samples, grouped_n], 0.0f0), name="qx_constituent_loc"))

            latent_vars[x_ng] = edmodels.PointMass(
                tf.Variable(tf.fill([species_num_samples, nongrouped_n], log(1.0f0/nongrouped_n)), name="qx_ng_loc"))
        end
    end

    # Inference
    # ---------

    if model_name == "pca"
        inference= ed.MAP(latent_vars, data)
        optimizer = tf.train[:AdamOptimizer](0.05)
        init_feed_dict = reduce(merge, [ls.init_feed_dict for ls in species_loaded_samples])
        Polee.run_inference(init_feed_dict, inference, 5000, optimizer)

        sess = ed.get_session()
        z_est = sess[:run](latent_vars[vars[:z]][:mean]())
        open("latent-ortholog-pca.csv", "w") do output
            print(output, "sample,")
            println(output, join([string("pc", j) for j in 1:num_components], ','))
            i = 1
            for k in 1:length(species_loaded_samples)
                ls = species_loaded_samples[k]
                for name in ls.sample_names

                    # TODO:
                    if match(r"(\d+)-\d+$", name).captures[1] == baseline_treatment
                        continue
                    end

                    # for j in 1:size(x_est, 2)
                    #     println(output, name, ",", j, ",", x_est[i, j])
                    # end
                    print(output, '"', name, '"', ',')
                    println(output, join(z_est[i,:], ','))
                    i += 1
                end
            end
        end
    elseif model_name == "regression"
        inference= ed.KLqp(latent_vars, data)
        optimizer = tf.train[:AdamOptimizer](0.05)

        init_feed_dict = reduce(merge, [ls.init_feed_dict for ls in species_loaded_samples])
        Polee.run_inference(init_feed_dict, inference, 10000, optimizer)

        sess = ed.get_session()
        w_cond_mean  = sess[:run](latent_vars[vars[:w_cond]][:mean]())
        w_cond_lower = sess[:run](latent_vars[vars[:w_cond]][:quantile](0.05))
        w_cond_upper = sess[:run](latent_vars[vars[:w_cond]][:quantile](0.95))
        open("latent-ortholog-regression.csv", "w") do output
            println(output, "ortholog_group,factor,lower_credible,mean,upper_credible")

            for j in 1:num_orth_groups
                for (factor, i) in cond_factors
                    println(
                        output, ortholog_group_names[j], ",",
                        factor, ",",
                        w_cond_lower[i, j], ",",
                        w_cond_mean[i, j], ",",
                        w_cond_upper[i, j])
                end
            end
        end
    end
end


"""
Build a factor matrix for species by parsing sample names.

(Obviously depends on a specific sample name format)
"""
function species_factor_matrix(species_loaded_samples, num_samples)
        species_code_to_idx = Dict{String, Int}()
        species_factor_weight = Dict{Int, Float32}()
        J = Int[]
        for k in 1:length(species_loaded_samples)
            ls = species_loaded_samples[k]
            n = size(ls.x0_values, 2)
            c = log(1.0f0/n)
            for name in ls.sample_names
                species_code = match(r"^(..)", name).captures[1]
                idx = get!(species_code_to_idx, species_code, length(species_code_to_idx) + 1)
                species_factor_weight[idx] = c
                push!(J, idx)
            end
        end

        factor_mat = zeros(Float32, (num_samples, length(species_code_to_idx)))
        for (i, j) in enumerate(J)
            factor_mat[i, j] = 1.0f0
        end

        return species_code_to_idx, factor_mat, species_factor_weight
end


function species_stage_factor_matrix(species_loaded_samples, num_samples)
        species_code_to_idx = Dict{String, Int}()
        species_factor_weight = Dict{Int, Float32}()
        J = Int[]
        for k in 1:length(species_loaded_samples)
            ls = species_loaded_samples[k]
            n = size(ls.x0_values, 2)
            c = log(1.0f0/n)
            for name in ls.sample_names
                species_code = match(r"^(..(:?-[AE])?)", name).captures[1]
                idx = get!(species_code_to_idx, species_code, length(species_code_to_idx) + 1)
                species_factor_weight[idx] = c
                push!(J, idx)
            end
        end

        factor_mat = zeros(Float32, (num_samples, length(species_code_to_idx)))
        for (i, j) in enumerate(J)
            factor_mat[i, j] = 1.0f0
        end

        return species_code_to_idx, factor_mat, species_factor_weight
end


function orthgroup_expr_initial_values(
    species_loaded_samples, species_transcripts,
    ortholog_group_by_id, num_samples, num_groups)

    x_og0 = zeros(Float32, (num_samples, num_groups))
    i = 1
    for (ls, ts) in zip(species_loaded_samples, species_transcripts)
        for k in 1:length(ls.sample_names)
            for (l, t) in enumerate(ts)
                if haskey(ortholog_group_by_id, t.metadata.name)
                    j = ortholog_group_by_id[t.metadata.name]
                    x_og0[i, j] += ls.x0_values[k, l]
                end
            end
            i += 1
        end
    end

    return log.(x_og0)
end


"""

"""
function species_condition_factor_matrix(
    species_loaded_samples, num_samples)

    species_factors = Dict{String, Int}()
    species_factor_weight = Dict{Int, Float32}()
    cond_factors = Dict{String, Int}()
    species_factor_idxs = Tuple{Int, Int}[]
    cond_factor_idxs = Tuple{Int, Int}[]
    k = 1
    for ls in species_loaded_samples
        n = size(ls.x0_values, 2)
        c = log(1.0f0/n)

        for name in ls.sample_names
            mat = match(r"^((..)(?:-(\D))?-(\d))", name)
            cond      = mat.captures[1]
            species   = mat.captures[2]
            stage     = mat.captures[3]
            treatment = mat.captures[4]

            l = get!(species_factors, string(species, "-", stage), 1 + length(species_factors))
            species_factor_weight[l] = c
            push!(species_factor_idxs, (k, l))

            if treatment != baseline_treatment
                push!(cond_factor_idxs,
                        (k, get!(cond_factors, cond, 1 + length(cond_factors))))
            end

            k += 1
        end
    end

    species_factor_mat = zeros(Float32, (num_samples, length(species_factors)))
    for (i, j) in species_factor_idxs
        species_factor_mat[i, j] = 1.0f0
    end

    cond_factor_mat = zeros(Float32, (num_samples, length(cond_factors)))
    for (i, j) in cond_factor_idxs
        cond_factor_mat[i, j] = 1.0f0
    end

    return species_factors, species_factor_mat, species_factor_weight, cond_factors, cond_factor_mat
end


function orthogroup_pca(
    species_loaded_samples, species_transcripts, ortholog_group_by_id,
     num_species, num_samples, num_groups, num_components=2)

    x0 = orthgroup_expr_initial_values(
        species_loaded_samples, species_transcripts,
        ortholog_group_by_id, num_samples, num_groups)

    # species_factor_mat, species_factor_weight =
    #     species_factor_matrix(species_loaded_samples, num_samples)
    species_factors, species_factor_mat, species_factor_weight =
        species_stage_factor_matrix(species_loaded_samples, num_samples)

    # compute a reasonable prior mean
    w_species_loc0 = Array{Float32}(length(species_factors), num_groups)
    for i in 1:size(w_species_loc0, 1)
        c = species_factor_weight[i]
        for j in 1:size(w_species_loc0, 2)
            # TODO: if we want uniform prior over transcripts, we need to
            # multiply this by the number of species i transcripts in group j.
            w_species_loc0[i, j] = c
        end
    end

    # TODO: option to toggle between different bias term schemes
    # species specific bias
    w_bias = edmodels.Normal(loc=tf.constant(w_species_loc0), scale=10.0f0)
    x_mu_bias = tf.matmul(species_factor_mat, w_bias)

    # fixed bias
    # x_mu_bias = edmodels.Normal(loc=tf.zeros([1, num_groups]), scale=10.0f0)

    # pca

    # map non-baseline samples to samples
    nonbaseline_sample_idxs = Int[]
    idx = 1
    for k in 1:length(species_loaded_samples)
        ls = species_loaded_samples[k]
        for name in ls.sample_names
            if match(r"(\d+)-\d+$", name).captures[1] != baseline_treatment
                push!(nonbaseline_sample_idxs, idx)
            end
            idx += 1
        end
    end
    num_nonbaseline_samples = length(nonbaseline_sample_idxs)

    nonbaseline_samples_mat = zeros(Float32, (num_samples, num_nonbaseline_samples))
    for (j, i) in enumerate(nonbaseline_sample_idxs)
        nonbaseline_samples_mat[i, j] = 1.0f0
    end

    w = edmodels.Normal(loc=tf.zeros([num_components, num_groups]),
                        scale=tf.fill([num_components, num_groups], 1.0f0))

    z = edmodels.Normal(loc=tf.zeros([num_nonbaseline_samples, num_components]),
                        scale=tf.fill([num_nonbaseline_samples, num_components], 1.0f0))

    x_mu_pca = tf.matmul(tf.constant(nonbaseline_samples_mat), tf.matmul(z, w))

    x_err_sigma_alpha0 = tf.constant(Polee.SIGMA_ALPHA0, shape=[1, num_groups])
    x_err_sigma_beta0 = tf.constant(Polee.SIGMA_BETA0, shape=[1, num_groups])
    x_err_sigma = edmodels.InverseGamma(x_err_sigma_alpha0, x_err_sigma_beta0, name="x_err_sigma")

    x_err = edmodels.Normal(
        loc=tf.fill([num_samples, num_groups], 0.0f0),
        scale=tf.sqrt(x_err_sigma), name="x_err")
    # x_err = edmodels.Normal(
    #     loc=tf.fill([num_samples, num_groups], 0.0f0),
    #     scale=tf.fill([num_samples, num_groups], 1.0f0), name="x_err")

    # TODO: why does including the x_err term make things so much worse?
    # x = x_mu_bias + x_mu_pca + x_err
    x = x_mu_bias + x_mu_pca

    # reasonable initialization for w_species
    nonbaseline_sample_idxs_set = Set(nonbaseline_sample_idxs)
    w_species0 = zeros(Float32, (length(species_factors), num_groups))
    w_species_count = zeros(Int, length(species_factors))
    for (i, j) in zip(findnz(species_factor_mat)[1:2]...)
        if baseline_treatment != nothing && i ∈ nonbaseline_sample_idxs_set
            continue
        end

        for k in 1:num_groups
            w_species0[j, k] += x0[i, k]
        end
        w_species_count[j] += 1
    end
    for i in 1:size(w_species_loc0, 1), j in 1:size(w_species_loc0, 2)
        w_species_loc0[i, j] /= w_species_count[i]
    end
    for i in 1:size(w_species0, 1), j in 1:size(w_species0, 2)
        w_species0[i, j] /= w_species_count[i]
    end

    latent_vars = Dict(
        # x_mu_bias => edmodels.PointMass(tf.Variable(tf.fill([1, num_groups], log(1.0f0/num_groups)))),
        w_bias => edmodels.PointMass(
            tf.Variable(w_species0)),
            # tf.Variable(tf.fill([num_species, num_groups], log(1.0f0/num_groups)), name="qw_bias_loc")),
        w => edmodels.PointMass(
            tf.Variable(tf.random_normal([num_components, num_groups]), name="qw_loc")),
            # tf.Variable(tf.zeros([num_components, num_groups]), name="qw_loc")),
        z => edmodels.PointMass(
            tf.Variable(tf.random_normal([num_nonbaseline_samples, num_components]), name="qz_loc")),
            # tf.Variable(tf.zeros([num_samples, num_components]), name="qz_loc")),
        x_err_sigma => edmodels.PointMass(tf.nn[:softplus](tf.Variable(tf.fill([1, num_groups], -3.0f0), name="qx_err_sigma"))),
        x_err => edmodels.PointMass(
            tf.Variable(tf.zeros([num_samples, num_groups]), name="qx_err_loc")))

    vars = Dict(
        # :w_bias   => w_bias,
        :w           => w,
        :z           => z,
        :x_err_sigma => x_err_sigma,
        :x_err       => x_err
    )

    return x, vars, latent_vars, nonbaseline_samples_mat
end


function orthogroup_regression(
    species_loaded_samples, species_transcripts, ortholog_group_by_id,
    num_species, num_samples, num_groups)

    x0 = orthgroup_expr_initial_values(
        species_loaded_samples, species_transcripts,
        ortholog_group_by_id, num_samples, num_groups)

    species_factors, species_factor_mat, species_factor_weight, cond_factors, cond_factor_mat =
        species_condition_factor_matrix(species_loaded_samples, num_samples)

    # compute a reasonable prior mean
    w_species_loc0 = Array{Float32}(length(species_factors), num_groups)
    for i in 1:size(w_species_loc0, 1)
        c = species_factor_weight[i]
        for j in 1:size(w_species_loc0, 2)
            w_species_loc0[i, j] = c
        end
    end

    w_species = edmodels.Normal(
        # loc=tf.fill([length(species_factors), num_groups], log(1.0f0/num_groups)),
        loc=tf.constant(w_species_loc0),
        scale=10.0f0)
    x_mu_species = tf.matmul(tf.constant(species_factor_mat), w_species)

    w_cond = edmodels.Normal(loc=tf.zeros([length(cond_factors), num_groups]),
                        scale=tf.fill([length(cond_factors), num_groups], 1.0f0))
    x_mu_cond = tf.matmul(tf.constant(cond_factor_mat), w_cond)

    x_err_sigma_alpha0 = tf.constant(Polee.SIGMA_ALPHA0, shape=[1, num_groups])
    x_err_sigma_beta0 = tf.constant(Polee.SIGMA_BETA0, shape=[1, num_groups])
    x_err_sigma = edmodels.InverseGamma(x_err_sigma_alpha0, x_err_sigma_beta0, name="x_err_sigma")
    x_err = edmodels.Normal(
        loc=tf.fill([num_samples, num_groups], 0.0f0),
        scale=tf.sqrt(x_err_sigma), name="x_err")

    x = x_mu_species + x_mu_cond + x_err

    # find a reasonable initial bias
    w_species0 = zeros(Float32, (length(species_factors), num_groups))
    w_species_count = zeros(Int, length(species_factors))
    for (i, j) in zip(findnz(species_factor_mat)[1:2]...)
        for k in 1:num_groups
            w_species0[j, k] += x0[i, k]
        end
        w_species_count[j] += 1
    end
    for i in 1:size(w_species_loc0, 1), j in 1:size(w_species_loc0, 2)
        w_species_loc0[i, j] /= w_species_count[i]
    end
    for i in 1:size(w_species0, 1), j in 1:size(w_species0, 2)
        w_species0[i, j] /= w_species_count[i]
    end

    latent_vars = Dict(
        w_species => edmodels.NormalWithSoftplusScale(
            loc=tf.Variable(w_species0, name="qw_species_loc"),
            scale=tf.Variable(tf.fill([length(species_factors), num_groups], 0.0f0), name="qw_species_scale")),
        w_cond => edmodels.NormalWithSoftplusScale(
            loc=tf.Variable(tf.zeros([length(cond_factors), num_groups]), name="qw_cond_loc"),
            scale=tf.Variable(tf.fill([length(cond_factors), num_groups], 0.0f0), name="qw_cond_scale")),
        x_err_sigma => edmodels.TransformedDistribution(
            distribution=edmodels.NormalWithSoftplusScale(
                loc=tf.Variable(tf.fill([1, num_groups], -1.0f0), name="qx_err_sigma_loc"),
                scale=tf.Variable(tf.fill([1, num_groups], 0.0f0), name="qx_err_sigma_scale")),
            bijector=tfdist.bijectors[:Softplus](),
            name="qx_err_sigma_sq"),
        x_err => edmodels.NormalWithSoftplusScale(
            loc=tf.Variable(tf.zeros([num_samples, num_groups]), name="qx_err_loc"),
            scale=tf.Variable(tf.fill([num_samples, num_groups], 0.0f0), name="qx_err_softplus_scale")))

    vars = Dict(
        :w_species   => w_species,
        :w_cond      => w_cond,
        :x_err_sigma => x_err_sigma,
        :x_err       => x_err
    )

    return x, vars, latent_vars, species_factors, cond_factors
end


main()
