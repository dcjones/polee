
# This implements to Kumaraswamy hierarchical stick breaking transform,
# including a heuristic for choosing the tree structure.


# Node in the completed tree
type HClustNode
    # 0 if internal node, >1 for child nodes giving the transcript index
    j::UInt32

    # self-loop used to indicate no children
    left_child::HClustNode
    right_child::HClustNode
    parent::HClustNode
    parent_idx::Int32

    subtree_size::Int32

    input_value::Float64
    grad::Float32
    ladj_grad::Float32

    k::Int

    function (::Type{HClustNode})(j::Integer)
        node = new(j)
        node.left_child = node
        node.right_child = node
        node.parent = node
        node.parent_idx = 0
        return node
    end

    function (::Type{HClustNode})(j::Integer, left_child::HClustNode,
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
    j1::Int
    j2::Int
    d::Float64
end


function Base.isless(a::HClustEdge, b::HClustEdge)
    return a.d < b.d
end


"""
Comparator used by delete! to force minheap items to the top.
"""
struct DeleteComparator
end


function DataStructures.compare(c::DeleteComparator, x, y)
    return true
end


"""
Delete and return the entry at index i.
"""
function Base.delete!(queue::MutableBinaryHeap, i::Int)
    nodes = queue.nodes
    nodemap = queue.node_map

    nd_id = nodemap[i]
    v0 = nodes[nd_id].value
    DataStructures._heap_bubble_up!(DeleteComparator(), nodes, nodemap, nd_id)
    return pop!(queue)
end


"""
Compute euclidean distance between two columns of the sparse matrix.
"""
function distance(X::SparseMatrixCSC, j1::Int, j2::Int)
    cp1 = Int(X.colptr[j1])
    cp2 = Int(X.colptr[j2])

    d = 0.0
    while cp1 < X.colptr[j1+1] && cp2 < X.colptr[j2+1]
        if X.rowval[cp1] < X.rowval[cp2]
            d += X.nzval[cp1]^2
            cp1 += 1
        elseif X.rowval[cp1] > X.rowval[cp2]
            d += X.nzval[cp2]^2
            cp2 += 1
        else
            d += (X.nzval[cp1] - X.nzval[cp2])^2
            cp1 += 1
            cp2 += 1
        end
    end

    while cp1 < X.colptr[j1+1]
        d += X.nzval[cp1]^2
        cp1 += 1
    end

    while cp2 < X.colptr[j2+1]
        d += X.nzval[cp2]^2
        cp2 += 1
    end

    # @show (Int(X.colptr[j1]), Int(X.colptr[j1+1]), Int(X.colptr[j2]), Int(X.colptr[j2+1]), cp1, cp2)
    # @show d

    # we inject a little noise to break ties randomly and lead to a more
    # balanced tree
    noise = 1e-20 * rand()
    return d + noise
end


function hclust(X::SparseMatrixCSC)
    println("Clustering transcripts")
    tic()

    m, n = size(X)

    # compare this many neighbors to each neighbors left and right to find
    # candidates to merge
    K = 10

    # order nodes by median compatible read index to group transcripts that
    # tend to share a lot of reads
    medreadidx = Array{UInt32}(n)
    for j in 1:n
        if X.colptr[j] == X.colptr[j+1]
            medreadidx[j] = 0
        else
            medreadidx[j] = X.rowval[div(X.colptr[j] + X.colptr[j+1], 2)]
        end
    end
    idxs = (1:n)[sortperm(medreadidx)]

    # initial edges
    queue = mutable_binary_minheap(HClustEdge)
    queue_idxs = Dict{Tuple{Int, Int}, Int}()
    neighbors = MultiDict{Int, Int}()
    set_size = Dict{Int, Int}()
    nodes = Dict{Int, HClustNode}()
    max_d = 0.0
    for j1 in 1:n
        for j2 in j1+1:min(j1+K, n)
            d = distance(X, idxs[j1], idxs[j2])
            max_d = max(max_d, d)
            e = HClustEdge(j1, j2, d)
            queue_idxs[(e.j1, e.j2)] = push!(queue, e)
            insert!(neighbors, j1, j2)
            insert!(neighbors, j2, j1)
        end
        set_size[j1] = 1
        nodes[j1] = HClustNode(idxs[j1])
    end
    deleted_nodes = Set{Int}()

    # cluster
    next_node_idx = n + 1
    while !isempty(queue)
        e = pop!(queue)
        delete!(queue_idxs, (e.j1, e.j2))

        # introduce a new node
        k = next_node_idx
        next_node_idx += 1

        for (ja, jb) in [(e.j1, e.j2), (e.j2, e.j1)]
            s_ja = set_size[ja]
            s_jb = set_size[jb]

            for l in neighbors[ja]
                if l == jb || l ∈ deleted_nodes
                    continue
                end
                # delete old edge
                u1, u2 = min((l, ja), (ja, l))
                old_edge = delete!(queue, queue_idxs[(u1, u2)])
                delete!(queue_idxs, (u1, u2))

                # already added this edge
                if haskey(queue_idxs, (l, k))
                    continue
                end

                # estimate (l, k) distance
                d_lja = old_edge.d

                v1, v2 = min((l, jb), (jb, l))
                if haskey(queue_idxs, (v1, v2))
                    d_ljb = queue[queue_idxs[(v1, v2)]].d
                else
                    d_ljb = max_d
                end

                d = (s_ja * d_lja + s_jb * d_ljb) / (s_ja + s_jb)

                # add (l, k) edge
                new_edge = HClustEdge(l, k, d)
                set_size[k] = s_ja + s_jb
                queue_idxs[(l, k)] = push!(queue, new_edge)
                insert!(neighbors, l, k)
                insert!(neighbors, k, l)
            end
        end

        nodes[k] = HClustNode(nodes[e.j1], nodes[e.j2])

        delete!(neighbors, e.j1)
        delete!(neighbors, e.j2)

        delete!(set_size, e.j1)
        delete!(set_size, e.j2)
        push!(deleted_nodes, e.j1)
        push!(deleted_nodes, e.j2)
    end

    toc()

    root = nodes[next_node_idx-1]
    return root
end


# Hierarchical stick breaking transformation
mutable struct HSBTransform
    # tree nodes in depth first traversal order
    nodes::Vector{HClustNode}

    # hint at an appropriate initial value for x. When cluster heuristic is used,
    # this is the approximate mode, otherwis its fill(1/n)
    x0::Vector{Float32}
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


function HSBTransform(X::SparseMatrixCSC, method::Symbol=:cluster)
    m, n = size(X)
    if method == :cluster
        root = hclust(X)
        x0 = fill(1.0f0/n, n)
        nodes = order_nodes(root, n)
    elseif method == :random
        nodes = rand_tree_nodes(n)
        x0 = fill(1.0f0/n, n)
    elseif method == :sequential
        nodes = rand_list_nodes(n)
        x0 = fill(1.0f0/n, n)
    else
        error("$(method) is not a supported HSB heuristic")
    end
    return HSBTransform(nodes, x0)
end


# rebuild tree from serialization
function HSBTransform{T<:Integer}(parent_idxs::Vector{T}, js::Vector{T})
    n = div(length(js) + 1, 2)
    return HSBTransform(deserialize_tree(parent_idxs, js), fill(1.0f0/n, n))
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
function flattened_tree(t::HSBTransform)
    return flattened_tree(t.nodes)
end


function flattened_tree(nodes::Vector{HClustNode})
    node_parent_idxs = Array{Int32}(length(nodes))
    node_js          = Array{Int32}(length(nodes))
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


function rand_hsb_tree(n)
    return HSBTransform(rand_tree_nodes(n))
end


function hsb_transform!{GRADONLY}(t::HSBTransform, ys::Vector, xs::Vector,
                                  ::Type{Val{GRADONLY}})
    nodes = t.nodes
    nodes[1].input_value = 1.0f0
    k = 1 # internal node count
    ladj = 0.0f0
    for i in 1:length(nodes)
        node = nodes[i]

        if node.j != 0 # leaf node
            xs[node.j] = node.input_value
            continue
        end

        @assert node.left_child !== node
        @assert node.right_child !== node

        node.k = k # TODO: this is dumb, I'm just keeping note of k for debugging purposes
        node.left_child.input_value = ys[k] * node.input_value
        node.right_child.input_value = (1 - ys[k]) * node.input_value

        if !GRADONLY
            # log jacobian determinant term for stick breaking
            ladj += log(node.input_value)
        end

        k += 1
    end
    @assert k == length(ys) + 1
    @assert isfinite(ladj)

    return ladj
end


function hsb_inverse_transform!(t::HSBTransform, xs::Vector, ys::Vector)
    nodes = t.nodes
    n = length(xs)
    k = n - 1 # internal node count
    for i in length(nodes):-1:1
        node = nodes[i]

        if node.j != 0 # leaf node
            node.input_value = xs[node.j]
            continue
        end

        node.input_value = node.left_child.input_value + node.right_child.input_value
        ys[k] = node.left_child.input_value / node.input_value

        k -= 1
    end
end


function hsb_transform_gradients!(t::HSBTransform, ys::Vector,
                                  y_grad::Vector,
                                  x_grad::Vector)
    nodes = t.nodes
    k = length(y_grad)
    for i in length(nodes):-1:1
        node = nodes[i]

        if node.j != 0 # leaf node
            node.grad = x_grad[node.j]
            node.ladj_grad = 0.0f0
        else
            # get derivative wrt y by multiplying children's derivatives by y's
            # contribution to their input values
            y_grad[k] = node.input_value *
                ((node.left_child.grad + node.left_child.ladj_grad) -
                 (node.right_child.grad + node.right_child.ladj_grad))

            # store derivative wrt this nodes input_value
            node.grad = ys[k] * node.left_child.grad + (1 - ys[k]) * node.right_child.grad

            # store ladj derivative wrt to input_value
            node.ladj_grad = 1/node.input_value +
                ys[k] * node.left_child.ladj_grad + (1 - ys[k]) * node.right_child.ladj_grad

            k -= 1
        end

    end
    @assert k == 0
end


"""
This are used by the tenorflow implementation of inverse HSB. It's faster to
build these matrices in julia than in python, so it's done here.
"""
function inverse_hsb_matrices(node_parent_idxs, node_js)

    # matrices with fewer than this many entries get merged into the adjacent
    # one. The tradeoff is that merged matrices use more memory and involve
    # redundant computation, but reduce overhead.
    MERGE_THRESHOLD = 20000

    num_nodes = length(node_parent_idxs)
    n = div(num_nodes + 1, 2)

    left_child = fill(Int32(-1), num_nodes)
    right_child = fill(Int32(-1), num_nodes)
    for i in 2:num_nodes
        parent_idx = node_parent_idxs[i]
        if right_child[parent_idx] == -1
            right_child[parent_idx] = i
        else
            left_child[parent_idx] = i
        end
    end

    # TODO: pass this along so we don't have to compute it in python
    leaf_indexes = zeros(Int32, n)
    internal_node_indexes = zeros(Int32, n-1)
    internal_node_left_indexes = zeros(Int32, n-1)
    k = 1
    for i in 1:num_nodes
        if node_js[i] != 0
            leaf_indexes[node_js[i]] = i
        else
            internal_node_indexes[k] = i
            internal_node_left_indexes[k] = left_child[i]
            k += 1
        end
    end

    q = Queue(Tuple{Int, Int})
    I = Int32[]
    J = Int32[]
    IJs = Tuple{Vector{Int32}, Vector{Int32}}[]
    last_height = 1
    enqueue!(q, (last_height, 1))
    while !isempty(q)
        if front(q)[1] != last_height
            push!(IJs, (I, J))
            I = Int[]
            J = Int[]
            last_height += 1
        end
        height, i = dequeue!(q)
        @assert height == last_height

        if left_child[i] != -1
            j = left_child[i]
            push!(I, i-1)
            push!(J, j-1)
            enqueue!(q, (height+1, j))
        end

        if right_child[i] != -1
            j = right_child[i]
            push!(I, i-1)
            push!(J, j-1)
            enqueue!(q, (height+1, j))
        end
    end

    if !isempty(I)
        push!(IJs, (I, J))
    end

    # post-process, merging very small matrices
    As = PyObject[]
    k = length(IJs)
    while k > 0
        if length(IJs[k][1]) < MERGE_THRESHOLD && k > 1
            I, J = merge_inverse_hsb_matrices(IJs[k-1][1], IJs[k-1][2], IJs[k][1], IJs[k][2])
            IJs[k-1] = (I, J)
        else
            I, J = IJs[k]
            push!(As, PyObject((tf.constant(I), tf.constant(J))))
        end
        k -= 1
    end

    @show length(As)

    return PyObject(
        (As, PyObject(leaf_indexes .- Int32(1)),
         PyObject(internal_node_indexes .- Int32(1)),
         PyObject(internal_node_left_indexes .- Int32(1))))
end


"""
Build some index arrays needed in the tensorflow implementation of inverse hsb.
"""
function make_inverse_hsb_params(node_parent_idxs, node_js)
    num_nodes = length(node_parent_idxs)
    n = div(num_nodes + 1, 2)

    # build leaf indexes
    leafindex = Array{Int32}(n)
    node_index_to_leaf_index = zeros(Int32, num_nodes)
    k = 1
    for i in 1:num_nodes
        if node_js[i] != 0
            leafindex[k] = node_js[i]
            node_index_to_leaf_index[i] = k
            k += 1
        end
    end

    # build internal node left sibling indexes
    left_child = fill(Int32(-1), num_nodes)
    right_child = fill(Int32(-1), num_nodes)
    for i in 2:num_nodes
        parent_idx = node_parent_idxs[i]
        if right_child[parent_idx] == -1
            right_child[parent_idx] = i
        else
            left_child[parent_idx] = i
        end
    end

    internal_node_indexes = zeros(Int32, n-1)
    internal_node_left_indexes = zeros(Int32, n-1)
    k = 1
    for i in 1:num_nodes
        if node_js[i] == 0
            internal_node_indexes[k] = i
            internal_node_left_indexes[k] = left_child[i]
            k += 1
        end
    end

    # need leftmost, rightmost, and node_js with all the zeros removed.
    leftmost  = fill(Int32(num_nodes+1), num_nodes)
    rightmost = fill(Int32(0), num_nodes)

    for i in 1:num_nodes
        if node_js[i] != 0
            leftmost[i] = i
            rightmost[i] = i
        end
    end

    for i in num_nodes:-1:2
        @assert leftmost[i] > 0
        @assert rightmost[i] > 0
        leftmost[node_parent_idxs[i]] =
            min(leftmost[node_parent_idxs[i]], leftmost[i])
        rightmost[node_parent_idxs[i]] =
            max(rightmost[node_parent_idxs[i]], rightmost[i])
    end

    # update indexes to index into a length n array of lead nodes
    for i in 1:num_nodes
        leftmost[i] = node_index_to_leaf_index[leftmost[i]]
        rightmost[i] = node_index_to_leaf_index[rightmost[i]]
        @assert leftmost[i] > 0
        @assert rightmost[i] > 0
    end

    @assert leftmost[1] == 1
    @assert rightmost[1] == n

    # make 0-based for tensorflow
    return (leafindex .- Int32(1),
            internal_node_indexes .- Int32(1),
            internal_node_left_indexes .- Int32(1),
            leftmost .- Int32(1),
            rightmost)
end


"""
If (Ia, Ja) and (Ib, Jb) represent two inverse hsb matrices, merge (Ib, Jb)
into (Ia, Ja) return a new result (Ic, Jc).
"""
function merge_inverse_hsb_matrices(Ia, Ja, Ib, Jb)
    p = sortperm(Ib)
    Ib = Ib[p]
    Jb = Jb[p]

    Ic = Int[]
    Jc = Int[]

    for (ia, ja) in zip(Ia, Ja)
        ks = searchsorted(Ib, ja)
        if isempty(ks)
            push!(Ic, ia)
            push!(Jc, ja)
        else
            for k in ks
                push!(Ic, ia)
                push!(Jc, Jb[k])
            end
        end
    end

    # We also need to include everything in Ib, Jb
    for (ib, jb) in zip(Ib, Jb)
        push!(Ic, ib)
        push!(Jc, jb)
    end

    return (Ic, Jc)
end
