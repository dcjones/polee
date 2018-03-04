
using HDF5
using PyCall

using DataStructures
include("../stick_breaking.jl")

@pyimport tensorflow as tf


logit(x) = log(x) - log(1 - x)

function main()
    inverse_hsb_op_module = tf.load_op_library("./inverse_hsb_op.so")

    input = h5open(ARGS[1])
    n = read(input["n"])
    node_parent_idxs = read(input["node_parent_idxs"])
    node_js = read(input["node_js"])
    t = HSBTransform(node_parent_idxs, node_js)
    num_nodes = 2*n-1

    # arrays needed by tensorflow op
    left_child = fill(Int32(-1), num_nodes)
    right_child = fill(Int32(-1), num_nodes)
    for i in 2:num_nodes
        parent_idx = node_parent_idxs[i]
        if right_child[parent_idx] == -1
            right_child[parent_idx] = i - 1
        else
            left_child[parent_idx] = i - 1
        end
    end
    leaf_index = Int32[j-1 for j in node_js]

    # Testing
    xs = rand(Float32, n)
    clamp!(xs, eps(Float32), 1 - eps(Float32))
    xs ./= sum(xs)

    ys_true = Array{Float32}(n-1)
    hsb_inverse_transform!(t, xs, ys_true)
    ys_logit_true = logit.(ys_true)
    # @show extrema(xs)
    # @show extrema(ys_true)
    # @show extrema(ys_logit_true)

    sess = tf.InteractiveSession()
    ys = inverse_hsb_op_module[:inv_hsb](
        tf.expand_dims(tf.constant(xs), 0),
        tf.expand_dims(tf.constant(left_child), 0),
        tf.expand_dims(tf.constant(right_child), 0),
        tf.expand_dims(tf.constant(leaf_index), 0))
    ys_logit_tf = sess[:run](ys)[1,:]

    # @show extrema(ys_logit_tf)
    @show extrema(abs.(ys_logit_tf .- ys_logit_true))

    # idx = indmax(abs.(ys_logit_tf .- ys_logit_true))
    # @show (idx, ys_logit_true[idx], ys_logit_tf[idx])

    # @show ys_logit_true[1:10]
    # @show ys_logit_tf[1:10]
end

main()


