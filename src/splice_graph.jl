

@enum SpliceGraphEdgeType TSSEdge ExonEdge IntronEdge TTSEdge


# What do we do about reverse strand transcripts?

"""
One edge in the splice graph.

`from` and `to` are 1-based inclusive exonic coordinates.
"""
struct SpliceGraphEdge
    from::Int
    to::Int
    type::SpliceGraphEdgeType
end

# What is actual graph representation?


function append_transcript_path!(
        edges::Dict{SpliceGraphEdge, Set{Int}},
        t::Transcript)
    exons = t.metadata.exons
    @assert !isempty(exons)
    id = t.metadata.id
    pushedge! = edge -> push!(get!(() -> Set{Int}(), edges, edge), id)

    if t.strand == STRAND_POS
        pushedge!(SpliceGraphEdge(-1, exons[1].first, TSSEdge))
        for i in 1:length(exons)
            pushedge!(SpliceGraphEdge(exons[i].first, exons[i].last, ExonEdge))
            if i < length(exons)
                pushedge!(SpliceGraphEdge(
                    exons[i].last, exons[i+1].first, IntronEdge))
            end
        end
        pushedge!(SpliceGraphEdge(exons[end].last, -1, TTSEdge))
    else
        pushedge!(SpliceGraphEdge(-1, exons[end].last, TSSEdge))
        for i in length(exons):-1:1
            pushedge!(SpliceGraphEdge(exons[i].last, exons[i].first, ExonEdge))
            if i > 1
                pushedge!(SpliceGraphEdge(
                    exons[i].first, exons[i-1].last, IntronEdge))
            end
        end
        pushedge!(SpliceGraphEdge(exons[1].first, -1, TTSEdge))
    end

    return edges
end


function gene_splice_graph(ts::Vector{Transcript})
    edges = Dict{SpliceGraphEdge, Set{Int}}()
    for t in ts
        append_transcript_path!(edges, t)
    end
    return edges
end


"""
A feature (multi)graph is a splice graph in which we compress paths into
single edges. Edges in the newly formed multigraph are annotade with types
along their paths.
"""
struct FeatureGraphEdge
    from::Int
    to::Int
    types::Vector{SpliceGraphEdgeType}
    ids::Set{Int} # ids of transcripts compatibe with this edge
end


function gene_feature_graph(ts::Vector{Transcript})

    splice_edges = gene_splice_graph(ts)

    edge_by_from = Dict{Int, Vector{SpliceGraphEdge}}()
    edge_by_to = Dict{Int, Vector{SpliceGraphEdge}}()
    visited = Dict{SpliceGraphEdge, Bool}()
    stack = SpliceGraphEdge[]
    for edge in keys(splice_edges)
        if !haskey(edge_by_from, edge.from)
            edge_by_from[edge.from] = SpliceGraphEdge[]
        end
        push!(edge_by_from[edge.from], edge)

        if !haskey(edge_by_to, edge.to)
            edge_by_to[edge.to] = SpliceGraphEdge[]
        end
        push!(edge_by_to[edge.to], edge)
        visited[edge] = false

        if edge.type == TSSEdge
            push!(stack, edge)
        end
    end

    feature_edges = Set{FeatureGraphEdge}()

    path_to = -1
    path_from = -1
    path_types = SpliceGraphEdgeType[]
    id_set = Set{Int}()

    while !isempty(stack)
        first_edge = pop!(stack)

        # add edge to path
        path_from = first_edge.from
        path_to = first_edge.to
        path_types = [first_edge.type]
        id_set = splice_edges[first_edge]

        while true
            if path_types[end] == TTSEdge
                push!(feature_edges,
                    FeatureGraphEdge(path_from, path_to, path_types, id_set))
                break
            # last node has in-degree or out-degree > 1
            elseif length(edge_by_to[path_to]) > 1 || length(edge_by_from[path_to]) > 1

                push!(feature_edges,
                    FeatureGraphEdge(path_from, path_to, path_types, id_set))

                for edge in edge_by_from[path_to]
                    if !visited[edge]
                        visited[edge] = true
                        push!(stack, edge)
                    end
                end
                break
            else
                edge = edge_by_from[path_to][1]
                path_to = edge.to
                push!(path_types, edge.type)
            end
        end
    end

    # @show (length(splice_edges), length(feature_edges))

    return feature_edges
end


# Common edge types used to classify alt splicing events
const IEIedge = [IntronEdge, ExonEdge, IntronEdge]
const Iedge = [IntronEdge]
const EIedge = [ExonEdge, IntronEdge]
const IEedge = [IntronEdge, ExonEdge]


function classify_feature_edges(edges::Vector{FeatureGraphEdge})
    classes = fill(:miscellaneous, length(edges))

    for (i, edge) in enumerate(edges)
        if edge.types[end] == TTSEdge
            classes[i] = :alt_tts
        elseif edge.types[1] == TSSEdge
            classes[i] = :alt_tss
        end
    end

    if length(edges) == 2
        t1 = edges[1].types
        t2 = edges[2].types
        t1t2 = (t1, t2)

        if t1t2 == (IEIedge, Iedge)
            classes .= [:included_cassette_exon, :skipped_cassette_exon]
        elseif t1t2 == (Iedge, IEIedge)
            classes .= [:skipped_cassette_exon, :included_cassette_exon]
        end
    elseif length(edges) > 2
        if all([edge.types == EIedge for edge in edges])
            fill!(classes, :alt_donor)
        elseif all([edge.types == IEedge for edge in edges])
            fill!(classes, :alt_acceptor)
        end
    end

    return classes
end


struct FeatureMetadata
    seqname::String
    strand::Strand
    first::Int
    last::Int
    type::Symbol
end


struct TSSMetadata
    seqname::String
    strand::Strand
    position::Int
end


function append_features!(
        is::Vector{Int}, js::Vector{Int}, feature_metadata::Vector{FeatureMetadata},
        num_features::Int, ts::Vector{Transcript})

    feature_edges = gene_feature_graph(ts)

    edge_by_from = Dict{Int, Vector{FeatureGraphEdge}}()
    for edge in feature_edges
        push!(get!(() -> Vector{FeatureGraphEdge}(), edge_by_from, edge.from), edge)
    end

    # @show length(ts)
    # for t in ts
    #     @show t.metadata.name
    #     @show t.metadata.exons
    # end

    seqname = first(ts).seqname
    strand = first(ts).strand

    for (from, edges) in edge_by_from
        if length(edges) == 1
            continue
        end

        # try to classify the type of alternative event
        feature_types = classify_feature_edges(edges)

        for (i, edge) in enumerate(edges)
            push!(
                feature_metadata,
                FeatureMetadata(seqname, strand, edge.from, edge.to, feature_types[i]))

            for id in edge.ids
                push!(is, id)
                push!(js, num_features)
            end
            num_features += 1
        end
    end

    return num_features
end


"""
Generate indexes for two sparse matrices:
    one genes onto transcripts and another mapping splice features onto transcripts
"""
function transcript_feature_matrices(
        ts::Transcripts, ts_metadata::TranscriptsMetadata)

    # transcripts by tss
    transcripts_by_tss = Dict{TSSMetadata, Vector{Transcript}}()
    for t in ts
        tss = t.strand == STRAND_POS ? t.metadata.exons[1].first : t.metadata.exons[1].last
        key = TSSMetadata(t.seqname, t.strand, tss)

        if !haskey(transcripts_by_tss, key)
            transcripts_by_tss[key] = Transcript[]
        end
        push!(transcripts_by_tss[key], t)
    end

    # gene <-> isoform design matrix
    tss_is = Int[]
    tss_js = Int[]
    tss_metadata = TSSMetadata[]

    # isoform <-> splice feature design matrix
    feature_is = Int[]
    feature_js = Int[]
    feature_metadata = FeatureMetadata[]

    num_features = 1

    # for (gene_id, gene_ts) in transcripts_by_gene_id
    for (i, (tss, tss_ts)) in enumerate(transcripts_by_tss)
        num_features = append_features!(
            feature_is, feature_js, feature_metadata, num_features, tss_ts)

        for t in tss_ts
            push!(tss_is, t.metadata.id)
            push!(tss_js, i)
            push!(tss_metadata, tss)
        end
    end

    return (tss_is, tss_js, tss_metadata, length(transcripts_by_tss),
            feature_is, feature_js, feature_metadata, num_features)
end
