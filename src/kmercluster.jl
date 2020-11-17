
# Hierarchical clustering of sequences based on approximate jaccard index
# between approximate k-mer sets.


"""
Candidate edge used during hierarchical clustering.
"""
struct KmerSetEdge
    i1::UInt32
    i2::UInt32
    jaccard::Float32
end

function Base.isless(a::KmerSetEdge, b::KmerSetEdge)
    return a.jaccard < b.jaccard
end

function Base.isgreater(a::KmerSetEdge, b::KmerSetEdge)
    return a.jaccard > b.jaccard
end


"""
Node in the hierarchical clustering tree.
"""
struct KmerSetNode
    # 0 if internal node, >1 for child nodes giving the transcript index
    j::UInt32

    left::Union{Nothing, KmerSetNode}
    right::Union{Nothing, KmerSetNode}

    subtree_size::UInt32
    subtree_height::UInt32

    function KmerSetNode(j::Integer)
        return new(j, nothing, nothing, 1, 0)
    end

    function KmerSetNode(j::Integer, left::KmerSetNode, right::KmerSetNode)
        return new(
            j, left, right,
            1 + left.subtree_size + right.subtree_size,
            1 + max(left.subtree_height, right.subtree_height))
    end
end


function KmerSetNode(left::KmerSetNode, right::KmerSetNode)
    return KmerSetNode(0, left, right)
end


# These comparison operators used so we can use huffman tree heuristic
# once we hit jaccard index of 0.
function Base.isless(a::KmerSetNode, b::KmerSetNode)
    return a.subtree_height < b.subtree_height
end

function Base.isgreater(a::KmerSetNode, b::KmerSetNode)
    return a.subtree_height > b.subtree_height
end


"""
Put nodes in DFS order and record parent indexes.
"""
function order_nodes(root::KmerSetNode, n)
    nodes = KmerSetNode[]
    sizehint!(nodes, n)

    parent_idxs = Int32[]
    sizehint!(parent_idxs, 2*n-1)

    stack = Tuple{Int32, KmerSetNode}[(0, root)]
    while !isempty(stack)
        parent_idx, node = pop!(stack)
        push!(nodes, node)
        push!(parent_idxs, parent_idx)
        node_idx = length(nodes)

        if node.left !== nothing
            push!(stack, (node_idx, node.left))
        end

        if node.right !== nothing
            push!(stack, (node_idx, node.right))
        end
    end

    return (parent_idxs, nodes)
end


"""
Find initial candidate edges with non-zero approximate jaccard.
"""
function initialize_priority_queue(sketches::Vector{KmerSketch{H,K}}) where {H,K}
    edges = Set{Tuple{UInt32, UInt32}}()
    pqueue = BinaryMaxHeap{KmerSetEdge}()

    # Maintain a ordered tree for every minhash bin so we can quickly find
    # collisions for new nodes.
    sketch_indexes = SortedMultiDict{UInt64, UInt32}[]
    sizehint!(sketch_indexes, H)

    # TODO: This could be done in parallel across minhash bins
    n = length(sketches)
    prog = Progress(H, 0.25, "Finding candidate edges", 60)
    idx = [UInt32(i) for i in 1:n]
    for bin in 1:H
        sort!(idx, by=i -> sketches[i].minhash[bin])

        # skip over empty (zero) bins
        i = 1
        while i < n && sketches[idx[i]].minhash[bin] == 0
            i += 1
        end

        while i < n
            # find a chunk [i, j-1] of equal minhash bins
            j = i + 1
            while j < n && sketches[idx[i]].minhash[bin] == sketches[idx[j]].minhash[bin]
                j += 1
            end

            # add every pair in the chunk to the priority queue
            for p in i:j-1, q in p+1:j-1
                (u, v) = (idx[p], idx[q])
                if (u, v) ∉ edges
                    push!(pqueue, KmerSetEdge(u, v, approximate_jaccard(sketches[u], sketches[v])))
                    push!(edges, (u, v))
                end
            end

            i = j
        end

        # TODO: could implement B+trees which would be a faster than SortedMultiDict
        push!(sketch_indexes,
            SortedMultiDict{UInt64, UInt32}([
                (sketches[i].minhash[bin] => i) for i in idx]))

        next!(prog)
    end
    finish!(prog)

    return (pqueue, sketch_indexes)
end


"""
Heuristically build a polya tree transformation from a set of transcript
sequences.
"""
function build_polya_tree_transformation(ts_tree::Transcripts)
    ts = collect(ts_tree)
    n = length(ts)

    # Sketch transcript kmer sets
    t0 = time()
    println("Computing sequence k-mer sets...")
    sketches = [KmerSketch{KMER_CLUSTER_H, KMER_CLUSTER_K}() for _ in 1:n]
    Threads.@threads for i in 1:n
        sketch = sketches[i]
        for x in each(DNAMer{KMER_CLUSTER_K}, LongDNASeq(ts[i].metadata.seq))
            push!(sketch, min(x.fw, x.bw))
        end
    end
    println("done. (", time() - t0, "s)")

    # Do initial clustering by order of largest jaccard index first
    nodes = Vector{KmerSetNode}()
    sizehint!(nodes, 2*n-1)
    for i in 1:n
        push!(nodes, KmerSetNode(i))
    end
    sizehint!(sketches, 2*n-1)

    # when adding new edges, used to keep track of what's already been added
    added_neighbors = Set{UInt32}()

    active_nodes = Set{UInt32}(1:n)
    pqueue, sketch_indexes = initialize_priority_queue(sketches)
    last_print = -1
    while !isempty(pqueue)
        edge = pop!(pqueue)

        if length(active_nodes) % 10000 == 0 && last_print != length(active_nodes)
            println("active_nodes: $(length(active_nodes)), jaccard: $(edge.jaccard)")
            last_print = length(active_nodes)
        end

        if edge.i1 ∉ active_nodes || edge.i2 ∉ active_nodes
            continue
        end

        # new node
        node = KmerSetNode(nodes[edge.i1], nodes[edge.i2])
        push!(nodes, node)
        i = length(nodes) # index of new node

        # create corresponding union sketch
        push!(sketches, union(sketches[edge.i1], sketches[edge.i2]))

        @assert length(nodes) == length(sketches)

        # delete old node, add new one
        delete!(active_nodes, edge.i1)
        delete!(active_nodes, edge.i2)
        push!(active_nodes, i)

        # populate new candidate edges.
        empty!(added_neighbors)
        # TODO: could be done in parallel. Somewhat tricky to avoid redundantly
        # adding edges to pqueue though. Better bet would be to just replace
        # SortedMultiDict with a BTree.
        for bin in 1:KMER_CLUSTER_H
            if sketches[i].minhash[bin] == 0
                continue
            end

            sketch_index = sketch_indexes[bin]
            eqrange = searchequalrange(sketch_index, sketches[i].minhash[bin])
            for (_, j) in inclusive(sketch_index, eqrange)
                if j ∉ active_nodes || j ∈ added_neighbors
                    continue
                end

                u, v = min(i, j), max(i, j)
                push!(pqueue, KmerSetEdge(u, v, approximate_jaccard(sketches[u], sketches[v])))
                push!(added_neighbors, j)
            end

            # Add new node to the sketch index
            insert!(sketch_indexes[bin], sketches[i].minhash[bin], i)
        end
    end

    # The remaining nodes have approximately 0 pairwise jaccard index, so we
    # switch to a different heuristic to join them into a tree. Essentially
    # this is building huffman coding trees: it greedily joins tho shortest
    # trees to produce something roughly balanced.
    minheight_queue = BinaryMinHeap{KmerSetNode}()
    for i in active_nodes
        push!(minheight_queue, nodes[i])
    end

    while length(minheight_queue) > 1
        a = pop!(minheight_queue)
        b = pop!(minheight_queue)
        push!(minheight_queue, KmerSetNode(a, b))
    end

    @assert length(minheight_queue) == 1

    root = pop!(minheight_queue)

    parent_idxs, dfs_nodes = order_nodes(root, n)
    node_js = Int32[node.j for node in dfs_nodes]

    return parent_idxs, node_js
end


