import TensorKit: TensorMap

@non_differentiable save(file, name, object)
@non_differentiable load(file, name)
@non_differentiable hamiltonian(model)

tospace(A::AbstractTensorMap, V1::VectorSpace, V2::VectorSpace) = TensorMap(A.data, V1, V2)
function ChainRulesCore.rrule(::typeof(tospace), x::AbstractTensorMap, V1::VectorSpace, V2::VectorSpace)
    y = TensorMap(x.data, V1, V2)
    function back(dy)
        dx = TensorMap(dy.data, space(x))
        return NoTangent(), dx, NoTangent(), NoTangent()
    end
    return y, back
end
