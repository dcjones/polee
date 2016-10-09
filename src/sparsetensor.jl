
using TensorFlow
import TensorFlow: with_op_name, add_input, set_attr_list, NodeDescription

type SparseTensor <: TensorFlow.AbstractTensor
    indices::Tensor
    values::Tensor
    shape::Tensor

    function SparseTensor(indices, values, shape)
        # TODO: check dimensions
        return new(Tensor(indices), Tensor(values), Tensor(shape))
    end
end


function TensorFlow.get_op(t::SparseTensor)
    return get_op(t.values)
end


function TensorFlow.get_def_graph(t::SparseTensor)
    return get_def_graph(t.indices)
end


function Base.eltype(t::SparseTensor)
    return eltype(t.values)
end


function sparse_tensor_dense_matmul(sp_a::SparseTensor, b,
                                    adjoint_a::Bool=false, adjoint_b::Bool=false;
                                    name="SparseTensorDenseMatMul")
    local desc
    with_op_name(name) do
        desc = NodeDescription("SparseTensorDenseMatMul")
        add_input(desc, sp_a.indices)
        add_input(desc, sp_a.values)
        add_input(desc, sp_a.shape)
        add_input(desc, Tensor(b))
        desc["adjoint_a"] = adjoint_a
        desc["adjoint_b"] = adjoint_b
    end
    Tensor(Operation(desc))
end



####### test

#sess = TensorFlow.Session(TensorFlow.Graph())

## These have to be zero based. This is the correct order.
#indices = Int[
    #0 0
    #0 1
    #1 0
    #1 1 ]

#values = Float32[1, 2, 3, 4]

#shape = Int[2, 2]

#a = SparseTensor(indices, values, shape)
#b = Float32[
     #1 2
     #3 4]
#@show b

#@show Tensor(values)
#@show Tensor(shape)
#@show Tensor(values)

#y = sparse_tensor_dense_matmul(a, b)

#@show run(sess, y)



