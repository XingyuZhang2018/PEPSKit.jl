abstract type HamiltonianModel end

"""
    Heisenberg(Ni::Int,Nj::Int,Jx::T,Jy::T,Jz::T) where {T<:Real}
    
return a struct representing the `Ni`x`Nj` unit cell heisenberg model with couplings `Jz`, `Jx` and `Jy`
"""
struct Heisenberg <: HamiltonianModel
    Ni::Int
    Nj::Int
    Jx::Real
    Jy::Real
    Jz::Real
end
Heisenberg(Ni,Nj) = Heisenberg(Ni,Nj,1.0,1.0,1.0)

const Sx = TensorMap(ComplexF64[0 1; 1 0]/2, ℂ^2, ℂ^2)
const Sy = TensorMap(ComplexF64[0 -1im; 1im 0]/2, ℂ^2, ℂ^2)
const Sz = TensorMap(ComplexF64[1 0; 0 -1]/2, ℂ^2, ℂ^2)
"""
    hamiltonian(model::Heisenberg)

return the heisenberg hamiltonian for the `model` as a two-site operator.
"""
hamiltonian(model::Heisenberg) = model.Jx * Sx ⊗ Sx +
                                 model.Jy * Sy ⊗ Sy +
                                 model.Jz * Sz ⊗ Sz

