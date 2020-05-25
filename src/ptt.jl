

# Polya Tree Transforms


struct PolyaTreeTransform{T}
    # 4 by (2n-1) matrix storing four types of indexes, stored in one matrix
    # for better cache locality.
    #   - index[1,:] -> output index (0 if internal node)
    #   - index[2,:] -> left child index (0 if leaf)
    #   - index[3,:] -> right child index (0 if leaf)
    #   - index[4,:] -> parent index (0 if root)
    index::Matrix{Int32}

    # following two matrices don't define the transformation, but are
    # pre-allocated space used when computing the transformation, or its
    # gradients.

    # value node receives from its parent when applying the transform
    us::Vector{Float64}

    # 2 by (2n-1) matrix of gradient intermediate values, stored in one matrix
    # to improve locality.
    #   - gradients[1,:] -> intermediate gradients
    #   - gradients[2,:] -> intermediate gradients of log det(J)
    gradients::Matrix{T}
end


"""
Heuristically build a PTT based on a sparse data. The defaut method will try
to define a transform that is able to preserve correlation/anticorrelation
between dimensions in X.
"""
function PolyaTreeTransform(
        X::SparseMatrixCSC, method::Symbol=:cluster, tree_root_ref=nothing)
    m, n = size(X)
    if method == :cluster
        root = hclust(X)
        if tree_root_ref !== nothing
            push!(tree_root_ref,root)
        end
        nodes = order_nodes(root, n)
    elseif method == :random
        nodes = rand_tree_nodes(n)
    elseif method == :sequential
        nodes = rand_list_nodes(n)
    else
        error("$(method) is not a supported Polya tree transform heuristic")
    end
    return PolyaTreeTransform(nodes)
end


"""
Build PTT from output of heirarchical clustering.
"""
function PolyaTreeTransform(nodes::Vector{HClustNode})
    num_nodes = length(nodes)
    index     = Array{Int32}(undef, (4, num_nodes))
    us        = Array{Float64}(undef, num_nodes)
    gradients = Array{Float32}(undef, (2, num_nodes))

    node_map = IdDict{HClustNode, Int32}()
    for (i, node) in enumerate(nodes)
        node_map[node] = i
    end

    for (i, node) in enumerate(nodes)
        index[1, i] = node.j
        index[2, i] = node.left_child === node ? 0 : node_map[node.left_child]
        index[3, i] = node.right_child === node ? 0 : node_map[node.right_child]
        index[4, i] = node.parent_idx
    end

    return PolyaTreeTransform{Float32}(index, us, gradients)
end


# function serialize()
    # I guess we don't need such a function because we can just write
    # t.index[1,:] and t.index[4,:]
# end


"""
Build PTT from serialized representation of the tree.
"""
function PolyaTreeTransform(
        parent_idxs::Vector{Int32}, output_idxs::Vector{Int32})
    @assert length(parent_idxs) == length(output_idxs)
    num_nodes = length(parent_idxs)

    index     = zeros(Int32, (4, num_nodes))
    us        = Array{Float64}(undef, num_nodes)
    gradients = Array{Float32}(undef, (2, num_nodes))

    for i in 1:num_nodes
        index[1, i] = output_idxs[i]

        # the tree was serialized in DFS order, traversing the right branch
        # first, so we know if our parent's right pointer is not set, we are
        # it's right child, otherwise left.
        if parent_idxs[i] != 0
            if index[3, parent_idxs[i]] == 0
                index[3, parent_idxs[i]] = i
            else
                index[2, parent_idxs[i]] = i
            end
        end

        index[4, i] = parent_idxs[i]
    end

    return PolyaTreeTransform{Float32}(index, us, gradients)
end



"""
Transform an n-1 vector `ys` in a hypercube, to a n vector `xs` in a simplex
using an Polya Tree Transform t. If `compute_ladj` compute and return the
log absolute determinant of the jacobian matrix.
"""
function transform!(
        t::PolyaTreeTransform{T},
        ys::AbstractVector,
        xs::AbstractVector,
        ::Val{compute_ladj}=Val(false)) where {compute_ladj, T}
    ladj = zero(T)
    t.us[1] = one(T)
    k = 1 # internal node count
    num_nodes = size(t.index, 2)
    for i in 1:num_nodes
        # leaf node
        output_idx = t.index[1, i]
        if output_idx != 0
            xs[output_idx] = t.us[i]
            xs[output_idx] = max(xs[output_idx], 1e-16)
            continue
        end

        # internal node
        left_idx = t.index[2, i]
        right_idx = t.index[3, i]

        t.us[left_idx] = ys[k] * t.us[i]
        t.us[right_idx] = (1 - ys[k]) * t.us[i]

        if compute_ladj
            ladj += log(t.us[i])
        end

        k += 1
    end
    @assert k == length(ys) + 1
    @assert isfinite(ladj)

    return ladj
end


"""
Where `x_grad` contains gradients df(x)/dx, compute df(x)/dy + dlog det(|J|)/dy
where x = T(y) and J is the jacobian matrix for T.
"""
function transform_gradients!(
        t::PolyaTreeTransform{T},
        ys::AbstractVector,
        y_grad::AbstractVector,
        x_grad::AbstractVector) where {T}
    num_nodes = size(t.index, 2)
    n = div(num_nodes + 1, 2)
    k = n - 1 # internal node number
    for i in num_nodes:-1:1
        # leaf node
        output_idx = t.index[1, i]
        if output_idx != 0
            t.gradients[1, i] = x_grad[output_idx]
            t.gradients[2, i] = zero(T)
            continue
        end

        left_idx = t.index[2, i]
        right_idx = t.index[3, i]

        left_grad = t.gradients[1, left_idx]
        left_ladj_grad = t.gradients[2, left_idx]

        right_grad = t.gradients[1, right_idx]
        right_ladj_grad = t.gradients[2, right_idx]

        # get derivative wrt y by multiplying children's derivatives by y's
        # contribution to their input values
        y_grad[k] = t.us[i] *
            ((left_grad + left_ladj_grad) - (right_grad + right_ladj_grad))

        # store derivative wrt this nodes input_value
        t.gradients[1, i] =
            ys[k] * left_grad + (1 - ys[k]) * right_grad

        # store ladj derivative wrt to input_value
        t.gradients[2, i] =
            1/t.us[i] + ys[k] * left_ladj_grad + (1 - ys[k]) * right_ladj_grad

        k -= 1
    end
    @assert k == 0
end


"""
Like `transform_gradients!` but without the jacobian term. Where `x_grad`
contains gradients df(x)/dx, compute df(x)/dy
where x = T(y) and J is the jacobian matrix for T.
"""
function transform_gradients_no_ladj!(
        t::PolyaTreeTransform{T},
        ys::AbstractVector,
        y_grad::AbstractVector,
        x_grad::AbstractVector) where {T}
    num_nodes = size(t.index, 2)
    n = div(num_nodes + 1, 2)
    k = n - 1 # internal node number
    for i in num_nodes:-1:1
        # leaf node
        output_idx = t.index[1, i]
        if output_idx != 0
            t.gradients[1, i] = x_grad[output_idx]
            t.gradients[2, i] = zero(T)
            continue
        end

        left_idx = t.index[2, i]
        right_idx = t.index[3, i]

        left_grad = t.gradients[1, left_idx]
        right_grad = t.gradients[1, right_idx]

        # get derivative wrt y by multiplying children's derivatives by y's
        # contribution to their input values
        y_grad[k] = t.us[i] * (left_grad - right_grad)

        # store derivative wrt this nodes input_value
        t.gradients[1, i] =
            ys[k] * left_grad + (1 - ys[k]) * right_grad

        k -= 1
    end
    @assert k == 0
end


"""
Transform a n vector `xs` on a simplex, to a `n-1` vector `ys` on a hypercube.
"""
function inverse_transform!(
        t::PolyaTreeTransform,
        xs::AbstractVector,
        ys::AbstractVector{T}) where {T}
    num_nodes = size(t.index, 2)
    n = div(num_nodes + 1, 2)
    k = n - 1 # internal node number
    ladj = zero(T)
    for i in num_nodes:-1:1
        # leaf node
        output_idx = t.index[1, i]
        if output_idx != 0
            t.us[i] = xs[output_idx]
            continue
        end

        left_idx = t.index[2, i]
        right_idx = t.index[3, i]

        t.us[i] = t.us[left_idx] + t.us[right_idx]
        ladj -= log(Float32(t.us[i]))
        ys[k] = t.us[left_idx] / t.us[i]

        k -= 1
    end

    @assert k == 0
    return ladj
end


"""
Given a partial Polya tree serialized as parent pointers (node_parent_idxs)
and leaf node indexes (node_js), build a more complete tree description
consising of left and right child pointers.
"""
function make_inverse_ptt_params(node_parent_idxs, node_js)
    num_nodes = length(node_js)

    left_index = fill(Int32(-1), num_nodes)
    right_index = fill(Int32(-1), num_nodes)
    for i in 2:num_nodes
        parent_idx = node_parent_idxs[i]
        if right_index[parent_idx] == -1
            right_index[parent_idx] = i - 1
        else
            left_index[parent_idx] = i - 1
        end
    end
    leaf_index = Int32[j-1 for j in node_js]

    return (left_index, right_index, leaf_index)
end
