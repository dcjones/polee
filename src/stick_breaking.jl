
# This implements to Kumaraswamy hierarchical stick breaking transform,
# including a heuristic for choosing the tree structure.


# Node in the completed tree
type HClustNode
    # 0 if internal node, >1 for child nodes giving the transcript index
    j::Int32

    # self-loop used to indicate no children
    left_child::HClustNode
    right_child::HClustNode
    parent::HClustNode
    parent_idx::Int32

    subtree_size::Int32

    input_value::Float64
    grad::Float32
    ladj_grad::Float32

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


# Node in a linked list of intermediate subtrees trees that still need to be merged
type SubtreeListNode
    # sparse vector contaaining probabilities or averaged probabilities
    vs::Vector{Float32}
    is::Vector{Int32}
    root::HClustNode

    # used to mark nodes that have already been merged so they
    # can be skipped when encountered in the priority queue
    merged::Bool
    left::SubtreeListNode
    right::SubtreeListNode

    function (::Type{SubtreeListNode})(vs::Vector{Float32}, is::Vector{Int32},
                                  root::HClustNode)
        node = new(vs, is, root, false)
        node.left = node
        node.right = node
        return node
    end
end


function Base.isless(a::Tuple{Float32, SubtreeListNode, SubtreeListNode},
                     b::Tuple{Float32, SubtreeListNode, SubtreeListNode})
    return a[1] < b[1]
end


function merge!(a::SubtreeListNode, b::SubtreeListNode,
                merge_buffer_v::Vector{Float32}, merge_buffer_i::Vector{Int32},
                queue, queue_idxs, K)

    # if b has the larger array, let that be reused, and a's array be gced
    if length(b.vs) > length(a.vs)
        a.vs, b.vs = b.vs, a.vs
        a.is, b.is = b.is, a.is
    end
    alen = length(a.is)
    blen = length(b.is)

    if length(merge_buffer_v) < alen
        resize!(merge_buffer_v, alen)
        resize!(merge_buffer_i, alen)
    end
    copy!(merge_buffer_v, a.vs)
    copy!(merge_buffer_i, a.is)

    # figure out space needed for merge
    merged_size = 0
    i, j = 1, 1
    while i <= alen && j <= blen
        if merge_buffer_i[i] < b.is[j]
            merged_size += 1
            i += 1
        elseif merge_buffer_i[i] > b.is[j]
            merged_size += 1
            j += 1
        else
            merged_size += 1
            i += 1
            j += 1
        end
    end

    merged_size += alen - i + 1
    merged_size += blen - j + 1

    if alen < merged_size
        resize!(a.is, merged_size)
        resize!(a.vs, merged_size)
    end

    # merge
    i, j, k = 1, 1, 1
    while i <= alen || j <= blen
        if j > blen || (i <= alen && merge_buffer_i[i] < b.is[j])
            a.is[k] = merge_buffer_i[i]
            a.vs[k] = merge_buffer_v[i] / 2.0f0
            i += 1
            k += 1
        elseif i > alen || (j <= blen && merge_buffer_i[i] > b.is[j])
            a.is[k] = b.is[j]
            a.vs[k] = b.vs[j] / 2.0f0
            j += 1
            k += 1
        else
            a.is[k] = merge_buffer_i[i]
            a.vs[k] = (merge_buffer_v[i] + b.vs[j]) / 2.0f0
            i += 1
            j += 1
            k += 1
        end
    end

    @assert k == merged_size + 1

    @assert issorted(a.is)

    node = HClustNode(a.root, b.root)
    a.root.parent = node
    b.root.parent = node
    ab = SubtreeListNode(a.vs, a.is, node)

    promote!(queue, queue_idxs, a, K)
    promote!(queue, queue_idxs, b, K)

    # stick ab in a's place
    if a.left !== a
        a.left.right = ab
        ab.left = a.left
    end

    if a.right !== a
        a.right.left = ab
        ab.right = a.right
    end

    # excise b
    if b.left !== b
        if b.right !== b
            b.left.right = b.right
            b.right.left = b.left
        else
            b.left.right = b.left
        end
    elseif b.right !== b
        b.right.left = b.right
    end

    a.merged = true
    b.merged = true

    return ab
end


# squared difference between sparse vectors stored in a and b
function distance(a::SubtreeListNode, b::SubtreeListNode)
    d = 0.0f0
    i = 1
    j = 1
    while i <= length(a.vs) && j <= length(b.vs)
        if a.is[i] < b.is[j]
            d += 1
            i += 1
        elseif a.is[i] > b.is[j]
            d -= 1
            j += 1
        else
            has_shared = true
            d += (a.vs[i] - b.vs[j]) / (a.vs[i] + b.vs[j])
            i += 1
            j += 1
        end
    end

    d += length(a.vs) - i + 1
    d -= length(b.vs) - j + 1

    # we inject a little noise to break ties randomly and lead to a more
    # balanced tree
    noise = 1f-20 * rand()
    return Float32(abs(d) + noise)
end


# Set adjacent node distances to 0.0 so they get popped immediately
# TODO: We can't look up items up if we use heap, so what are we to do?
function promote!(queue, queue_idxs, a, K)
    if a.left !== a
        u = a.left
        for _ in 1:K
            if haskey(queue_idxs, (u, a))
                idx = queue_idxs[(u,a)]
                DataStructures.update!(queue, idx, (0.0f0, u, a))
            end
            if u.left === u
                break
            else
                u = u.left
            end
        end
    end

    if a.right !== a
        u = a.right
        for _ in 1:K
            if haskey(queue_idxs, (a, u))
                idx = queue_idxs[(a,u)]
                DataStructures.update!(queue, idx, (0.0f0, a, u))
            end
            if u.right === u
                break
            else
                u = u.right
            end
        end
    end
end


function update_distances!(queue, queue_idxs, a, K)
    if a.left !== a
        u = a.left
        for _ in 1:K
            d = distance(u, a)
            queue_idxs[(u,a)] = push!(queue, (d, u, a))
            if u.left === u
                break
            else
                u = u.left
            end
        end
    end

    if a.right !== a
        u = a.right
        for _ in 1:K
            d = distance(a, u)
            queue_idxs[(a,u)] = push!(queue, (d, a, u))
            if u.right === u
                break
            else
                u = u.right
            end
        end
    end
end


function hclust_initalize(X::SparseMatrixRSB, n, K)
    # construct nodes
    I, J, V = findnz(X)
    p = sortperm(J)
    I = I[p]
    J = J[p]
    V = V[p]

    tic()
    k = 1
    j = 1
    nodes = Array{SubtreeListNode}(n)
    while j <= n && k <= length(J)
        if j < J[k]
            nodes[j] = SubtreeListNode(Float32[], Int32[], HClustNode(j))
            j += 1
        elseif j > J[k]
            error("Error in clustering. This is a bug")
        else
            k0 = k
            k += 1
            while k <= length(J) && J[k] == j
                k += 1
            end
            nodes[j] = SubtreeListNode(V[k0:k-1], I[k0:k-1], HClustNode(j))
            @assert issorted(nodes[j].is)
            j += 1
        end
    end

    while j <= n
        nodes[j] = SubtreeListNode(Float32[], Int32[], HClustNode(j))
        j += 1
    end
    toc() # 0.6 seconds

    # order nodes by median compatible read index to group transcripts that
    # tend to share a lot of reads
    medreadidx = Array{Int32}(n)
    for (j, node) in enumerate(nodes)
        medreadidx[j] = isempty(nodes[j].is) ? 0 : nodes[j].is[div(length(nodes[j].is) + 1, 2)]
    end
    nodes = nodes[sortperm(medreadidx)]

    # turn nodes into a linked list
    for i in 2:n
        nodes[i].left = nodes[i-1]
        nodes[i-1].right = nodes[i]
    end

    tic()
    queue = mutable_binary_minheap(Tuple{Float32, SubtreeListNode, SubtreeListNode})
    queue_idxs = Dict{Tuple{SubtreeListNode, SubtreeListNode}, Int}()
    for i in 1:n
        for j in i+1:min(i+K, n)
            d = distance(nodes[i], nodes[j])
            queue_idxs[(nodes[i], nodes[j])] = push!(queue, (d, nodes[i], nodes[j]))
        end
    end
    toc() # 3 seconds

    return queue, queue_idxs
end


function hclust(X::SparseMatrixRSB)

    srand(102030310)

    m, n = size(X)
    # compare this many neighbors to each neighbors left and right to find
    # candidates to merge
    K = 5
    queue, queue_idxs = hclust_initalize(X, n, K)

    merge_buffer_v = Float32[]
    merge_buffer_i = Int32[]

    steps = 0
    merge_count = 0
    prog = Progress(n-1, 0.25, "Clustering transcripts ", 60)
    while true
        steps += 1
        d, a, b = pop!(queue)
        delete!(queue_idxs, (a,b))

        if a.merged || b.merged
            continue
        end

        # @show d

        ab = merge!(a, b, merge_buffer_v, merge_buffer_i, queue, queue_idxs, K)
        merge_count += 1
        next!(prog)

        # all trees have been merged
        if ab.left === ab && ab.right === ab
            return ab.root
        end

        update_distances!(queue, queue_idxs, ab, K)
    end
end


# Hierarchical stick breaking transformation
type HSBTransform
    # tree nodes in depth first traversal order
    nodes::Vector{HClustNode}
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

    check_subtree_sizes(nodes[1])
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

    return nodes
end


function HSBTransform(X::SparseMatrixRSB)
    m, n = size(X)
    root = hclust(X)
    nodes = order_nodes(root, n)
    return HSBTransform(nodes)
end


# rebuild tree from serialization
function HSBTransform(parent_idxs, js)
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

    return HSBTransform(nodes)
end


# generate a random HSBTransform tree with n leaf nodes.
function rand_hsb_tree(n)
    stack = [HClustNode(j) for j in 1:n]

    while length(stack) > 1
        # this is a ridiculously inefficient way of doing this, but we only
        # need this for small scale testing
        shuffle!(stack)
        a = pop!(stack)
        b = pop!(stack)
        push!(stack, HClustNode(a, b))
    end
    root = stack[1]

    nodes = order_nodes(root, n)
    return HSBTransform(nodes)
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


function hsb_transform_gradients!(t::HSBTransform, ys::Vector,
                                  y_grad::Vector,
                                  x_grad::Vector)
    nodes = t.nodes
    k = length(y_grad)
    for i in length(nodes):-1:1
        node = nodes[i]

        if node.j != 0 # leaf node
            node.grad = x_grad[node.j]
            node.ladj_grad = 1.0f0
        else
            yk = ys[k]

            # get derivative wrt y by multiplying children's derivatives by y's
            # contribution to their input values
            y_grad[k] = node.input_value *
                ((node.left_child.grad + node.left_child.ladj_grad) -
                 (node.right_child.grad + node.right_child.ladj_grad))

            # store derivative wrt this nodes input_value
            node.grad = ys[k] * node.left_child.grad + (1 - ys[k]) * node.right_child.grad

            # store derivative wrt to ladj
            node.ladj_grad = 1/node.input_value +
                ys[k] * node.left_child.ladj_grad + (1 - ys[k]) * node.right_child.ladj_grad

            k -= 1
        end

    end
    @assert k == 0
end

