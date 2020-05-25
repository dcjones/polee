
# This implements to Kumaraswamy hierarchical stick breaking transform,
# including a heuristic for choosing the tree structure.


# Node in the completed tree
mutable struct HClustNode
    # 0 if internal node, >1 for child nodes giving the transcript index
    j::UInt32

    # self-loop used to indicate no children
    left_child::HClustNode
    right_child::HClustNode
    parent::HClustNode
    parent_idx::Int32

    subtree_size::Int32
    read_count::UInt32
    child_jaccard::Float32

    # TODO: Would like to delete these values but still used by ilr
    input_value::Float64
    grad::Float32
    ladj_grad::Float32

    k::Int

    function HClustNode(j::Integer)
        node = new(j)
        node.left_child = node
        node.right_child = node
        node.parent = node
        node.parent_idx = 0
        return node
    end

    function HClustNode(j::Integer, left_child::HClustNode,
                                  right_child::HClustNode)
        node = new(j, left_child, right_child)
        node.parent = node
        node.parent_idx = 0
        return node
    end
end


function HClustNode(left_child::HClustNode, right_child::HClustNode)
    return HClustNode(0, left_child, right_child)
end


function maxdepth(root::HClustNode)
    return 1 +
        max(root.left_child === root ? 0 : maxdepth(root.left_child),
            root.right_child === root ? 0 : maxdepth(root.right_child))
end


function height(node::HClustNode)
    h = 1
    while node.parent !== node
        h += 1
        node = node.parent
    end
    return h
end


function write_node_heights(nodes::Vector{HClustNode})
    out = open("node-heights.csv", "w")
    for i in 1:length(nodes)
        node = nodes[i]
        if node.j != 0 # leaf node
            println(out, Int(node.j), ",", height(node))
        end
    end
    close(out)
end


struct HClustEdge
    j1::UInt32
    j2::UInt32
    similarity::Float32
end


function Base.isless(a::HClustEdge, b::HClustEdge)
    return a.similarity < b.similarity
end

function Base.isgreater(a::HClustEdge, b::HClustEdge)
    return a.similarity > b.similarity
end


struct NodeWithSize
    j::UInt32
    size::UInt32
end

function Base.isless(a::NodeWithSize, b::NodeWithSize)
    return a.size < b.size
end

function Base.isgreater(a::NodeWithSize, b::NodeWithSize)
    return a.size > b.size
end


function read_set_intersection_size(rs1::Vector{UInt32}, rs2::Vector{UInt32})
    if isempty(rs1) || isempty(rs2) || rs1[1] > rs2[end] || rs1[end] < rs2[1]
        return 0
    end

    i = 1
    j = 1
    intersection_size = 0
    while i <= length(rs1) && j <= length(rs2)
        if rs1[i] < rs2[j]
            i += 1
        elseif rs1[i] > rs2[j]
            j += 1
        else
            i += 1
            j += 1
            intersection_size += 1
        end
    end
    return intersection_size
end


function read_set_size_diff(rs1::Vector{UInt32}, rs2::Vector{UInt32})
    return -1 - abs(length(rs1) - length(rs2))
end


function read_set_relative_intersection_size(rs1::Vector{UInt32}, rs2::Vector{UInt32})
    if isempty(rs1) && isempty(rs2)
        return 0.0
    end

    intersection_size = read_set_intersection_size(rs1, rs2)
    union_size = length(rs1) + length(rs2) - intersection_size

    return intersection_size / union_size
end


function merge_read_sets(
        rs1::Vector{UInt32}, rs2::Vector{UInt32})
    intersection_size = read_set_intersection_size(rs1, rs2)
    rs_merged = Vector{UInt32}(
        undef, length(rs1) + length(rs2) - intersection_size)

    i = 1
    j = 1
    k = 1
    while i <= length(rs1) && j <= length(rs2)
        if rs1[i] < rs2[j]
            rs_merged[k] = rs1[i]
            i += 1
        elseif rs1[i] > rs2[j]
            rs_merged[k] = rs2[j]
            j += 1
        else
            rs_merged[k] = rs1[i]
            i += 1
            j += 1
        end
        k += 1
    end

    while i <= length(rs1)
        rs_merged[k] = rs1[i]
        i += 1
        k += 1
    end

    while j <= length(rs2)
        rs_merged[k] = rs2[j]
        j += 1
        k += 1
    end

    @assert length(rs_merged) == k - 1

    return rs_merged
end


function hclust(X::SparseMatrixCSC)
    println("Clustering transcripts")

    m, n = size(X)

    # compare this many neighbors to each neighbors left and right to find
    # candidates to merge
    K = 25

    # order nodes by median compatible read index to group transcripts that
    # tend to share a lot of reads
    medreadidx = Array{UInt32}(undef, n)
    for j in 1:n
        if X.colptr[j] == X.colptr[j+1]
            medreadidx[j] = 0
        else
            medreadidx[j] = X.rowval[div(X.colptr[j] + X.colptr[j+1], 2)]
        end
    end
    idxs = (1:n)[sortperm(medreadidx)]

    # add initial nodes
    read_sets = Dict{UInt32, Vector{UInt32}}()
    nodes = Dict{UInt32, HClustNode}()
    for j in 1:n
        nodes[j] = HClustNode(idxs[j])
        read_sets[j] = X.rowval[X.colptr[idxs[j]]:X.colptr[idxs[j]+1]-1]
        nodes[j].read_count = length(read_sets[j])
        nodes[j].child_jaccard = 1.0
    end

    # add initial edges
    queue = BinaryMaxHeap{HClustEdge}()
    neighbors = MultiDict{UInt32, UInt32}()
    for j1 in 1:n
        for j2 in j1+1:min(j1+K, n)
            similarity = read_set_relative_intersection_size(read_sets[j1], read_sets[j2])
            if similarity > 0
                push!(queue, HClustEdge(j1, j2, similarity))
            end
            insert!(neighbors, j1, j2)
            insert!(neighbors, j2, j1)
        end
    end

    # build tree greedily by joining the subtrees that share the most reads first
    next_node_idx = n + 1
    deleted_nodes = Set{UInt32}()

    next_node_idx = hclust_join_edges!(
        queue, nodes, deleted_nodes, read_sets, neighbors,
        next_node_idx, read_set_relative_intersection_size)
    @assert isempty(queue)

    remainder_queue = BinaryMinHeap{NodeWithSize}()
    for j in keys(nodes)
        # add one to the number of reads, subtree size plays a small role
        # and the tree is a bit more balanaced
        push!(remainder_queue, NodeWithSize(j, 1 + length(read_sets[j])))
    end
    while length(remainder_queue) > 1
        node1 = pop!(remainder_queue)
        node2 = pop!(remainder_queue)

        k = next_node_idx
        next_node_idx += 1
        nodes[k] = HClustNode(nodes[node1.j], nodes[node2.j])
        nodes[k].read_count = nodes[node1.j].read_count + nodes[node2.j].read_count
        nodes[k].child_jaccard = 0.0
        push!(remainder_queue, NodeWithSize(k, node1.size + node2.size))
    end

    @assert length(remainder_queue) == 1
    return nodes[pop!(remainder_queue).j]
end


function hclust_join_edges!(
        queue, nodes, deleted_nodes, read_sets, neighbors,
        next_node_idx, similarity_function)

    while !isempty(queue)
        e = pop!(queue)

        # stale edge
        if e.j1 ∈ deleted_nodes || e.j2 ∈ deleted_nodes
            continue
        end

        # new node
        k = next_node_idx
        next_node_idx += 1
        read_sets[k] = merge_read_sets(
            read_sets[e.j1], read_sets[e.j2])
        nodes[k] = HClustNode(nodes[e.j1], nodes[e.j2])
        nodes[k].read_count = length(read_sets[k])
        nodes[k].child_jaccard = e.similarity

        # delete old nodes
        delete!(nodes, e.j1)
        delete!(read_sets, e.j1)
        delete!(nodes, e.j2)
        delete!(read_sets, e.j2)
        push!(deleted_nodes, e.j1)
        push!(deleted_nodes, e.j2)

        # add new edges
        for (ja, jb) in ((e.j1, e.j2), (e.j2, e.j1))
            for l in neighbors[ja]
                if l == jb || l ∈ deleted_nodes
                    continue
                end

                similarity = similarity_function(read_sets[l], read_sets[k])

                if similarity != 0
                    push!(queue, HClustEdge(l, k, similarity))
                end

                insert!(neighbors, l, k)
                insert!(neighbors, k, l)
            end
        end
    end

    return next_node_idx
end


function set_subtree_sizes!(nodes::Vector{HClustNode})
    # traverse up the tree setting the subtree size
    for i in length(nodes):-1:1
        if nodes[i].j != 0
            nodes[i].subtree_size = 1
        else
            nodes[i].subtree_size = nodes[i].left_child.subtree_size +
                                    nodes[i].right_child.subtree_size
        end
    end

    # check_subtree_sizes(nodes[1])
end


function check_subtree_sizes(node::HClustNode)
    if node.j != 0
        @assert node.subtree_size == 1
        return 1
    else
        nl = check_subtree_sizes(node.left_child)
        nr = check_subtree_sizes(node.right_child)
        @assert node.subtree_size == nl + nr
        return nl + nr
    end
end


function check_node_ordering(nodes)
    for k in 2:length(nodes)
        node = nodes[k]
        @assert node.parent_idx < k
        @assert nodes[node.parent_idx].left_child === node ||
                nodes[node.parent_idx].right_child === node
    end
end


# Put nodes in DFS order and set parent_idx
function order_nodes(root::HClustNode, n)
    nodes = HClustNode[]
    sizehint!(nodes, n)
    stack = HClustNode[root]
    while !isempty(stack)
        node = pop!(stack)
        push!(nodes, node)

        if node.j == 0 # internal node
            @assert node.left_child !== node
            @assert node.right_child !== node

            node.left_child.parent_idx  = length(nodes)
            node.right_child.parent_idx = length(nodes)

            push!(stack, node.left_child)
            push!(stack, node.right_child)
        else
            @assert node.left_child === node
            @assert node.right_child === node
        end
    end

    # check_node_ordering(nodes)
    set_subtree_sizes!(nodes)
    # write_node_heights(nodes)
    # exit()
    return nodes
end


function deserialize_tree(parent_idxs, js)
    @assert length(parent_idxs) == length(js)
    nodes = [HClustNode(0) for i in 1:length(parent_idxs)]
    nodes[1].j = js[1]
    for i in 2:length(nodes)
        nodes[i].j = js[i]
        nodes[i].parent = nodes[parent_idxs[i]]
        nodes[i].parent_idx = parent_idxs[i]

        parent_node = nodes[parent_idxs[i]]

        # right branch is expanded first in the DFS order we use
        if parent_node.right_child === parent_node
            parent_node.right_child = nodes[i]
        else
            parent_node.left_child = nodes[i]
        end
    end

    set_subtree_sizes!(nodes)

    return nodes
end


"""
Reduce the tree structure to two arrays: on giving parent indexes, one giving
leaf indexes.
"""
function flattened_tree(nodes::Vector{HClustNode})
    node_parent_idxs = Array{Int32}(undef, length(nodes))
    node_js          = Array{Int32}(undef, length(nodes))
    for i in 1:length(nodes)
        node = nodes[i]
        node_parent_idxs[i] = node.parent_idx
        node_js[i] = node.j
    end

    return Dict{String, Vector}(
        "node_parent_idxs" => node_parent_idxs,
        "node_js"          => node_js)
end


"""
Generate a random HSB tree.
"""
function rand_tree_nodes(n)
    stack = [HClustNode(j) for j in 1:n]

    # TODO: This is insanely inefficient. Maybe that's ok since it just exists
    # as a straw man for comparison.
    while length(stack) > 1
        shuffle!(stack)
        a = pop!(stack)
        b = pop!(stack)
        push!(stack, HClustNode(a, b))
    end
    root = stack[1]

    nodes = order_nodes(root, n)
    return nodes
end


"""
Generate a random HSB transformation by building a list tree with nodes in a
random order. (Basically to mimic the stick breaking transformation in stan.)
"""
function rand_list_nodes(n)
    stack = [HClustNode(j) for j in 1:n]
    shuffle!(stack)
    while length(stack) > 1
        a = pop!(stack)
        b = pop!(stack)
        push!(stack, HClustNode(a, b))
    end

    root = stack[1]

    nodes = order_nodes(root, n)
    return nodes
end


# Some debugging utilities
function node_label(node)
    if node.j != 0
        return @sprintf("L%d", node.j)
    else
        return @sprintf("I%d", node.k)
    end
end


function show_node(node, xs, ys)
    if node.j != 0
        println(node_label(node), "[label=\"x=", xs[node.j], "\"];")
    else
        println(node_label(node), "[label=\"", node.k, "\\ny=", round(ys[node.k], 4), "\\nin=", node.input_value, "\"];")
        println(node_label(node), " -> ", node_label(node.left_child), ";")
        println(node_label(node), " -> ", node_label(node.right_child), ";")
        show_node(node.left_child, xs, ys)
        show_node(node.right_child, xs, ys)
    end
end
