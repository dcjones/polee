# Essentially this just draws a heatmap of splicing predictions of splicing
# features with the largest changes between conditions, but there are some
# additional tricks. Colors are desaturated towards grey with either high
# uncertainty or low gene expression.

using Colors, Compose, SQLite, Optim, Clustering

"""
This is loosely based on the idea of "value supressing uncertainty palettes.
When v is near 0, the value is fixed color (i.e. grey) regardless of the
value of u. So it converges to a single color along the v-axis, unlike a regular
bivariate color palette that would try to show two independent dimensions.
"""
function colormap(u, v)

    # lightness
    l_u_end = 80.0
    l_u_mid = 20.0
    l_u = l_u_mid + 2*abs(u - 0.5) * (l_u_end - l_u_mid)

    l_v = 80.0
    l = l_u * v + l_v * (1-v)

    # chroma
    c_min = 0.0
    c_max = 70.0
    c = c_min + v * (c_max - c_min)

    # hue
    h_min = 265.0 # blue
    h_max = 85.0  # yellow
    h = h_min + u * (h_max - h_min)
    # h = h_min + u * abs(h_max - h_min)

    LCHab(l, c, h)
end


# draw a color grid, for testing
if false
    ctx = context()
    nsteps = 30
    l = 1/nsteps
    for (i, u) in enumerate(range(0.0, stop=1.0, length=nsteps))
        for (j, v) in enumerate(range(0.0, stop=1.0, length=nsteps))
            x = (i - 1) / nsteps
            y = (j - 1) / nsteps
            compose!(ctx,
                (context(), rectangle(x, y, l, l), fill(f(u, v))))
        end
    end
    compose!(ctx, svgattribute("shape-rendering", "crispEdges"))

    ctx |> SVG("colors.svg", 4inch, 4inch)
    # ctx |> PNG("colors.png", 4inch, 4inch)
end


function row_distances(U, V)
    m = size(U, 1)
    D = zeros(Float64, (m, m))
    for i1 in 1:m, i2 in 1:m
        D[i1, i2] = sum((U[i1,:] .- U[i2,:]).^2) + sum((V[i1,:] .- V[i2,:]).^2)
        # D[i1, i2] = sum((V[i1,:] .- V[i2,:]).^2)
    end
    return D
end

function col_distances(U, V)
    n = size(U, 2)
    D = zeros(Float64, (n, n))
    for j1 in 1:n, j2 in 1:n
        D[j1, j2] = sum((U[:,j1] .- U[:,j2]).^2) + sum((V[:,j1] .- V[:,j2]).^2)
        # D[j1, j2] = sum((V[:,j1] .- V[:,j2]).^2)
    end
    return D
end


# function optimal_permutation(D)
#     n = size(D, 1)
#     @assert size(D, 2) == n

#     # make relative distance matrix
#     R = D ./ sum(D, dims=2)
#     R .+= 1e-6

#     # 1-D embedding
#     es0 = randn(n)

#     function objective(es)
#         De = sqrt.((es .- transpose(es)).^2)
#         Re = De ./ sum(De, dims=2)
#         Re .+= 1e-6
#         return sum(R .* log.(R./Re))
#     end

#     @show objective(es0)
#     result = optimize(objective, es0, LBFGS())
#     @show result

#     es_opt = Optim.minimizer(result)
#     return sortperm(es_opt)
# end

function optimal_permutation(D)
    result = hclust(D, linkage=:average)
    return result.order
end


logistic(x) = 1 / (1+exp(-x))


function get_top_splicing_features(db, feature_type, max_num_features)
    # find top features
    # SQLite.execute!(
    #     db,
    #     """
    #     create temporary table top_features as
    #     select distinct feature_num
    #     from sp_eff_db.effects
    #     join splicing_features
    #     on sp_eff_db.effects.splicing_num == splicing_features.feature_num
    #     where factor != "bias"
    #     and type == "$(feature_type)"
    #     order by abs(w) desc
    #     limit $(max_num_features)
    #     """)

    SQLite.execute!(
        db,
        """
        create temporary table top_features as
        select distinct splicing_num as feature_num, span
        from sp_eff_db.splicing_prop_span
        join splicing_features
        on sp_eff_db.splicing_prop_span.splicing_num == splicing_features.feature_num
        where type == "$(feature_type)"
        order by span desc
        limit $(max_num_features)
        """)

    @show collect(SQLite.Query(db, "select min(span) from top_features"))
    @show collect(SQLite.Query(db, "select max(span) from top_features"))

    SQLite.execute!(
        db,
        """
        create temporary table splicing_bias as
        select splicing_num, w as w0
        from sp_eff_db.effects
        where factor == "bias"
        """)

    # select coefficients for top features
    query = SQLite.Query(
        db,
        """
        select factor, w + w0 as y, feature_num
        from sp_eff_db.effects
        join top_features
          on sp_eff_db.effects.splicing_num == top_features.feature_num
        join splicing_bias
          on sp_eff_db.effects.splicing_num == splicing_bias.splicing_num
        where factor != "bias"
        """)

    row_index = Dict{Int, Int}()
    col_index = Dict{String, Int}()

    cols = Int[]
    rows = Int[]
    values = Float64[]

    for row in query
        if !haskey(col_index, row[:factor])
            col_index[row[:factor]] = 1+length(col_index)
        end

        if !haskey(row_index, row[:feature_num])
            row_index[row[:feature_num]] = 1+length(row_index)
        end

        push!(cols, col_index[row[:factor]])
        push!(rows, row_index[row[:feature_num]])
        push!(values, logistic(row[:y]))
    end

    return cols, rows, values, row_index, col_index
end


function get_splicing_uncertainty(db, row_index, col_index)
    query = SQLite.Query(
        db,
        """
        select *
        from splicing_prop_credible_interval
        join top_features
          on splicing_prop_credible_interval.splicing_num == top_features.feature_num
        """)

    cols = Int[]
    rows = Int[]
    values = Float64[]

    for row in query
        if row[:factor] == "bias"
            continue
        end

        i = row_index[row[:splicing_num]]
        j = col_index[row[:factor]]
        push!(rows, i)
        push!(cols, j)
        push!(values, row[:p_upper] - row[:p_lower])
    end

    return cols, rows, values
end


function get_transcript_regression_coefs(db, row_index, col_index)

    # compute factor-wise transcript expression for the transcripts involved
    # in the top features being plotted.

    SQLite.execute!(
        db,
        """
        create temporary table transcript_bias as
        select transcript_num, w as w0
        from tr_eff_db.effects
        where factor == "bias"
        """)

    transcript_expr = Dict{Tuple{String, Int}, Float64}() # (factor, transcript_num) -> expr

    for side in ["including", "excluding"]
        query = SQLite.Query(
            db,
            """
            select
            splicing_feature_$(side)_transcripts.transcript_num,
            factor,
            w + w0
            from splicing_feature_$(side)_transcripts
            join top_features
            on top_features.feature_num == splicing_feature_$(side)_transcripts.feature_num
            join tr_eff_db.effects
            on tr_eff_db.effects.transcript_num == splicing_feature_$(side)_transcripts.transcript_num
            join transcript_bias
            on transcript_bias.transcript_num == splicing_feature_$(side)_transcripts.transcript_num
            where factor != "bias"
            """)

        for (transcript_num, factor, log_expr) in query
            transcript_expr[(factor, transcript_num)] = log_expr
        end
    end

    involved_ids = Dict{Int, Vector{Int}}() # feaure_num -> [transcript_num]
    for side in ["including", "excluding"]
        query = SQLite.Query(
            db,
            """
            select
            splicing_feature_$(side)_transcripts.feature_num,
            splicing_feature_$(side)_transcripts.transcript_num
            from splicing_feature_$(side)_transcripts
            join top_features
            on top_features.feature_num == splicing_feature_$(side)_transcripts.feature_num
            """)
        for (feature_num, transcript_num) in query
            if !haskey(involved_ids, feature_num)
                involved_ids[feature_num] = Int[]
            end
            push!(involved_ids[feature_num], transcript_num)
        end
    end

    cols = Int[]
    rows = Int[]
    values = Float64[]

    for (feature_num, transcript_nums) in involved_ids
        i = row_index[feature_num]
        for (factor, j) in col_index
            expr = 0.0
            for transcript_num in transcript_nums
                expr += exp(transcript_expr[(factor, transcript_num)])
            end
            push!(rows, i)
            push!(cols, j)
            push!(values, log(expr))
        end
    end

    # TODO: Now that I think about this, I don't know that including the bias
    # was important, since we are going to normalise across rows anyway.

    return cols, rows, values
end

function drawplot(feature_type, num_features)
    db = SQLite.DB()
    SQLite.execute!(db, "attach \"genes.db\" as genes_db;")
    SQLite.execute!(db, "attach \"transcript-effects.db\" as tr_eff_db;")
    SQLite.execute!(db, "attach \"splicing-effects.db\" as sp_eff_db;")

    u_cols, u_rows, u_values, row_index, col_index =
        get_top_splicing_features(db, feature_type, num_features)

    v_cols, v_rows, v_values =
        get_splicing_uncertainty(db, row_index, col_index)

    # v_cols, v_rows, v_values =
    #     get_transcript_regression_coefs(db, row_index, col_index)

    # Now we need to build a color matrix
    m, n = length(row_index), length(col_index)
    U = zeros(Float64, (m, n))
    for (i, j, u) in zip(u_rows, u_cols, u_values)
        U[i, j] = u
    end

    V = zeros(Float64, (m, n))
    for (i, j, v) in zip(v_rows, v_cols, v_values)
        V[i, j] = v
    end

    # U_row_min = minimum(U, dims=2)
    # U_row_max = maximum(U, dims=2)
    # U = (U .- U_row_min) ./ (U_row_max .- U_row_min)

    # normalize rows
    # V_row_min = minimum(V, dims=2)
    # V_row_max = maximum(V, dims=2)
    # V = (V .- V_row_min) ./ (V_row_max .- V_row_min)

    # normalize accross the whole matrix
    # V_row_min = minimum(V)
    # V_row_max = maximum(V)
    # V = (V .- V_row_min) ./ (V_row_max .- V_row_min)

    @show minimum(V)
    @show maximum(V)

    # V[:,:] .= 1.0

    row_perm = optimal_permutation(row_distances(U, V))
    col_perm = optimal_permutation(col_distances(U, V))

    C = Array{LCHab}(undef, (m, n))
    for i in 1:m, j in 1:n
        C[i,j] = colormap(U[i,j], 1.0 - V[i,j])
    end

    C = C[row_perm,col_perm]

    long_labels = Dict(
        "tissue_Sp"  => "Spleen",
        "tissue_Te"  => "Testis",
        "tissue_He"  => "Heart",
        "tissue_Lin" => "Large intestine",
        "tissue_Br"  => "Brain",
        "tissue_Ag"  => "Adrenal gland",
        "tissue_Ut"  => "Uterus",
        "tissue_Th"  => "Thymus",
        "tissue_Lu"  => "Lung",
        "tissue_Mu"  => "Muscle",
        "tissue_St"  => "Stomach",
        "tissue_Fs"  => "Forestomach",
        "tissue_Sin" => "Small intestine",
        "tissue_Ov"  => "Ovary",
        "tissue_Ki"  => "Kidney",
        "tissue_Vg"  => "Vesicular gland",
        "tissue_Bm"  => "Bone marrow",
        "tissue_Li"  => "Liver"
    )

    labels = Vector{String}(undef, length(col_index))
    for (factor, j) in col_index
        labels[j] = long_labels[factor]
    end
    labels = labels[col_perm]

    label_width = max_text_extents("Quire Sans Pro", 10pt, labels...)[1]
    label_positions = collect(range(1/(2*n), stop=(2*n-1)/(2*n), length=n))
    colkey = compose(
        context(0, 0, 1w, label_width),
        font("Quire Sans Pro"),
        fontsize(10pt),
        text(
            label_positions, [1.0],
            labels, [hleft], [vcenter],
            [Rotation(-0.5pi, p, 1.0) for p in label_positions]))

    heatmap = context(0, label_width, 1w, 1h-label_width)
    for i in 1:m, j in 1:n
        x = (j-1)/n
        y = (i-1)/m
        compose!(
            heatmap,
            (context(), rectangle(x, y, 1/n, 1/m), fill(C[i,j])))
    end
    compose!(heatmap, svgattribute("shape-rendering", "crispEdges"))
    ctx = compose(context(), colkey, heatmap)
    ctx |> SVG("$(feature_type)-heatmap.svg", 4inch, 10inch)
    # colkey |> SVG("heatmap.svg", 4inch, 8inch)

    return U[row_perm, col_perm], V[row_perm, col_perm]
end

num_features = 100
feature_types = [
    "mutex_exon",
    "cassette_exon",
    "retained_intron",
    "alt_5p_end",
    "alt_3p_end",
    "alt_5p_end",
    "alt_3p_end",
    "alt_acceptor_site",
    "alt_donor_site" ]

for feature_type in feature_types
    drawplot(feature_type, num_features)
end
