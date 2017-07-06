
# This implements to Kumaraswamy hierarchical stick breaking transform,
# including a heuristic for choosing the tree structure.


# Node in the completed tree
type HClustNode
    # 0 if internal node, >1 for child nodes giving the transcript index
    j::Int32

    # self-loop used to indicate no children
    left_child::HClustNode
    right_child::HClustNode

    rightmost_path::Bool

    input_value::Float32
    grad::Float32

    function (::Type{HClustNode})(j::Integer)
        node = new(j)
        node.left_child = node
        node.right_child = node
        return node
    end

    function (::Type{HClustNode})(j::Integer, left_child::HClustNode,
                                  right_child::HClustNode)
        return new(j, left_child, right_child)
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


function merge!(a::SubtreeListNode, b::SubtreeListNode, queue, K)
    i = 1
    j = 1
    while i <= length(a.vs) && j <= length(b.vs)
        if a.is[i] < b.is[j]
            a.vs[i] /= 2.0f0
            i += 1
        elseif a.is[i] > b.is[j]
            a.is[i], b.is[j] = b.is[j], a.is[i]
            a.vs[i], b.vs[j] = b.vs[j], a.vs[i]
            i += 1
        else
            # TODO: if I'm averaging these, should every other value be divided
            # by zero. Maybe something to test emperically.
            a.vs[i] = (a.vs[i] + b.vs[j]) / 2.0f0
            i += 1
            j += 1
        end
    end

    if j <= length(b.vs)
        k = length(b.vs) - j + 1
        resize!(a.vs, length(a.vs) + k)
        # copy!(a.vs, i, b.vs, j, k)
        u = i
        for v in j:length(b.vs)
            a.vs[u] = b.vs[v] / 2.0f0
            u += 1
        end

        resize!(a.is, length(a.is) + k)
        copy!(a.is, i, b.is, j, k)
    end

    node = HClustNode(a.root, b.root)
    ab = SubtreeListNode(a.vs, a.is, node)

    promote!(queue, a, K)
    promote!(queue, b, K)

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
            i += 1
        elseif a.is[i] > b.is[j]
            j += 1
        else
            d += (a.vs[i] - b.vs[j])^2
            i += 1
            j += 1
        end
    end

    while i <= length(a.vs)
        d += a.vs[i]^2
        i += 1
    end

    while j <= length(b.vs)
        d += b.vs[j]^2
        j += 1
    end

    # we inject a little noise to break ties randomly and lead to a more
    # balanced tree
    noise = 1f-12 * rand()
    return d + noise
end


# Set adjacent node distances to Inf
function promote!(queue::PriorityQueue, a, K)
    if a.left !== a
        u = a.left
        for _ in 1:K
            if haskey(queue, (u, a))
                queue[(u, a)] = 0.0f0
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
            if haskey(queue, (a, u))
                queue[(a, u)] = 0.0f0
            end
            if u.right === u
                break
            else
                u = u.right
            end
        end
    end
end


function update_distances!(queue::PriorityQueue, a, K)
    if a.left !== a
        u = a.left
        for _ in 1:K
            queue[(u, a)] = distance(u, a)
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
            queue[(a, u)] = distance(a, u)
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
    # shuffle first so nodes with the same median read idx are in random order,
    # leading to a more balanced tree on average (TODO: does this actually help?)
    shuffle!(nodes)
    nodes = nodes[sortperm(medreadidx)]

    # @show medreadidx[1:20]

    # turn nodes into a linked list
    for i in 2:n
        nodes[i].left = nodes[i-1]
        nodes[i-1].right = nodes[i]
    end

    tic()
    queue = PriorityQueue(Tuple{SubtreeListNode, SubtreeListNode}, Float32)
    for i in 1:n
        for j in i+1:min(i+K, n)
            queue[(nodes[i], nodes[j])] = distance(nodes[i], nodes[j])
        end
    end
    toc() # 3 seconds

    return queue
end


function hclust(X::SparseMatrixRSB)
    m, n = size(X)
    # compare this many neighbors to each neighbors left and right to find
    # candidates to merge
    K = 5
    queue = hclust_initalize(X, n, K)

    # TODO: most of the time is spend on queue maintanence. We could do better
    # by using a basic binary heap and periodically walking through and
    # promoting all merged node comparisons so they get purged.

    # tic()
    steps = 0
    merge_count = 0
    # Profile.start_timer()
    while true
        steps += 1
        a, b = dequeue!(queue)

        if a.merged || b.merged
            continue
        end

        ab = merge!(a, b, queue, K)
        merge_count += 1

        # all trees have been merged
        if ab.left === ab && ab.right === ab
            # @show (n, steps, merge_count)
            # toc()
            # Profile.stop_timer()
            # Profile.print()
            return ab.root
        end

        update_distances!(queue, ab, K)
    end
end


# Hierarchical stick breaking transformation
type HSBTransform
    # tree nodes in depth first traversal order
    nodes::Vector{HClustNode}
end


# Put nodes in DFS order and label rightmost path
function order_nodes(root::HClustNode, n)
    nodes = HClustNode[]
    sizehint!(nodes, n)
    root.rightmost_path = true
    stack = HClustNode[root]
    while !isempty(stack)
        node = pop!(stack)
        push!(nodes, node)

        if node.j == 0 # internal node
            @assert node.left_child !== node
            @assert node.right_child !== node

            node.left_child.rightmost_path = false
            push!(stack, node.left_child)

            node.right_child.rightmost_path = node.rightmost_path
            push!(stack, node.right_child)
        end
    end
    return nodes
end


function HSBTransform(X::SparseMatrixRSB)
    m, n = size(X)
    root = hclust(X)
    nodes = order_nodes(root, n)
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


function hsb_transform!(t::HSBTransform, ys::Vector{Float32}, xs::Vector{Float32})
    nodes = t.nodes
    nodes[1].input_value = 1.0f0
    k = 1 # internal node count
    for i in 1:length(nodes)
        node = nodes[i]

        if node.j != 0 # leaf node
            xs[node.j] = node.input_value
            continue
        end

        @assert 0.0f0 <= node.input_value <= 1.0f0
        @assert 0.0f0 <= ys[k] <= 1.0f0
        @assert node.left_child !== node
        @assert node.right_child !== node

        node.left_child.input_value = ys[k] * node.input_value
        node.right_child.input_value = (1.0f0 - ys[k]) * node.input_value

        k += 1
    end
    @assert k == length(ys) + 1

    # Log absolute jacobian determinant calculation. Like the transform itself
    # this is computed with a simple tree traversal, but we need to specialy
    # handle every node on the path from right-most leaf to root.
    ladj = 0.0f0
    k = 1
    for i in 1:length(nodes)
        node = nodes[i]
        if node.j != 0 # leaf node
            continue
        end

        ladj += log(node.input_value)
        if node.rightmost_path
            ladj += log(1 - ys[k])
        end
        k += 1
    end

    return ladj
end


function hsb_gradients!(t::HSBTransform, y_grad::Vector{Float32}, x_grad::Vector{Float32})
    nodes = t.nodes
    k = length(y_grad)
    for i in length(nodes):-1:1
        node = nodes[i]

        if node.j != 0 # leaf node
            node.grad = x_grad[node.j]
        else
            # get derivative wrt y by multiplying children's derivatives by y's
            # contribution to their input values
            y_grad[k] = node.input_value * (node.left_child.grad - node.right_child.grad)

            # store derivative wrt this nodes input_value
            node.grad = ys[k] * node.left_child.grad + (1 - ys[k]) * node.right_child.grad
            k -= 1
        end
    end
    @assert k == 0

    # gradients of log absolute jacobian determinant
    k = length(y_grad)
    for i in length(nodes):-1:1
        node = nodes[i]

        if node.j != 0 # leaf node
            node.grad = 1.0f0
        else
            # derivative of the ladj in this subtree wrt to y[k] which is
            # derived by multipying children's gradient's by contribution to
            # their input values
            y_grad[k] += node.input_value * (node.left_child.grad - node.right_child.grad)
            if node.rightmost_path
                y_grad[k] += 1 / (ys[k] - 1)
            end

            # store derivative of the ladj wrt to this node's input_value
            node.grad = 1/node.input_value +
                ys[k] * node.left_child.grad + (1 - ys[k]) * node.right_child.grad
            k -= 1
        end
    end
    @assert k == 0
end

