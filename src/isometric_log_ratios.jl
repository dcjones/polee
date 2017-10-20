
# isometric log-ratio transformation
type ILRTransform
    # tree nodes in depth first traversal order
    nodes::Vector{HClustNode}
    xs_sum::Float64
end


function flattened_tree(t::ILRTransform)
    return flattened_tree(t.nodes)
end


function ILRTransform(nodes::Vector{HClustNode})
    return ILRTransform(nodes, 0.0)
end


function ILRTransform(X::SparseMatrixCSC, method::Symbol=:cluster)
    m, n = size(X)
    root, xs0 = hclust(X)
    nodes = order_nodes(root, n)
    return ILRTransform(nodes)
end


function ILRTransform(parent_idxs::Vector, js::Vector)
    nodes = deserialize_tree(parent_idxs, js)
    return ILRTransform(nodes)
end


function rand_ilr_tree(n)
    return ILRTransform(rand_tree_nodes(n))
end


"""
Transfrom real numbers ys to simplex constrain vector xs. (This is actually
the inverse ilr transformation, if we follew the literature.)
"""
function ilr_transform!{GRADONLY}(t::ILRTransform, ys::Vector, xs::Vector,
                                  ::Type{Val{GRADONLY}})

    nodes = t.nodes
    nodes[1].input_value = 0.0f0
    k = 1 # internal node count
    ladj = 0.0

    for i in 1:length(nodes)
        node = nodes[i]

        if node.j != 0 # leaf node
            xs[node.j] = exp(node.input_value)
            continue
        end

        @assert node.left_child !== node
        @assert node.right_child !== node

        r = Int(node.left_child.subtree_size)
        s = Int(node.right_child.subtree_size)

        a =  sqrt(s / (r*(r+s)))
        b = -sqrt(r / (s*(r+s)))

        node.left_child.input_value = a * ys[k] + node.input_value
        node.right_child.input_value = b * ys[k] + node.input_value

        k += 1
    end

    t.xs_sum = sum(xs) # stored to use when computing gradients
    xs ./= t.xs_sum

    if !GRADONLY
        n = length(xs)
        for x in xs
            ladj += log(x)
        end
        ladj += log(sqrt(n))
    end

    return ladj
end

"""
Compute ILR transform gradients.
"""
function ilr_transform_gradients!(t::ILRTransform, xs::Vector, y_grad::Vector,
                                  x_grad::Vector)

    # constant used in gradients wrt x prior to normalization
    xs_sum = t.xs_sum
    c = 0.0
    for i in 1:length(xs)
        dladj_dxi = 1 / xs[i]
        c += xs[i] * (x_grad[i] + dladj_dxi) / xs_sum
    end

    nodes = t.nodes
    k = length(y_grad)
    for i in length(nodes):-1:1
        node = nodes[i]

        if node.j != 0 # leaf node
            dladj_dxi = 1 / xs[node.j]
            node.grad = (1/xs_sum) * (x_grad[node.j] + dladj_dxi) - c   # normalization
            node.grad = exp(node.input_value) * node.grad # exp transforation
        else
            r = Int(node.left_child.subtree_size)
            s = Int(node.right_child.subtree_size)

            a =  sqrt(s / (r*(r+s)))
            b = -sqrt(r / (s*(r+s)))

            left_grad = node.left_child.grad
            right_grad = node.right_child.grad

            node.grad = left_grad + right_grad
            y_grad[k] = a * left_grad + b * right_grad

            k -= 1
        end
    end
    @assert k == 0
end


"""
Transorm simplex constrained vector xs to unconstrained real numbers ys.

This is only for testing purposes, so it doesn't compute the jacobian determinant.
"""
function inverse_ilr_transform!{GRADONLY}(t::ILRTransform, xs::Vector, ys::Vector,
                                          ::Type{Val{GRADONLY}})

    nodes = t.nodes
    k = length(ys)
    for i in length(nodes):-1:1
        node = nodes[i]

        if node.j != 0 # leaf node
            node.input_value = log(xs[node.j])
        else
            r = Int(node.left_child.subtree_size)
            s = Int(node.right_child.subtree_size)

            a =  sqrt(s / (r*(r+s)))
            b = -sqrt(r / (s*(r+s)))

            left_value = node.left_child.input_value
            right_value = node.right_child.input_value

            ys[k] = a * left_value + b * right_value

            node.input_value = left_value + right_value

            k -= 1
        end
    end

    @assert k == 0
end


