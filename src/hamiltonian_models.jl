abstract type HamiltonianModel end

"""
    Heisenberg(Ni::Int,Nj::Int,Jx::Real,Jy::Real,Jz::Real) 
    
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


"""
    FreeFermion(Ni::Int,Nj::Int) 
    
return a struct representing the `Ni`x`Nj` unit cell FreeFermion model 
"""
struct FreeFermion <: HamiltonianModel
    Ni::Int
    Nj::Int
    t::Real
end
FreeFermion(Ni,Nj) = FreeFermion(Ni,Nj,1.0)

"""
    hamiltonian_array(model::FreeFermion)
    
return the FreeFermion hamiltonian for the `model` as a two-site operator.
"""
function hamiltonian_array(model::FreeFermion)
    t = model.t
    H = zeros(4,4,4,4)

    H[1,4,4,1], H[1,2,4,3], H[4,1,1,4], H[4,3,1,2] = -t, -t, -t, -t
    H[1,3,3,1], H[4,3,2,1], H[3,1,1,3], H[2,1,4,3] = -t, -t, -t, -t
    H[3,4,2,1], H[3,2,2,3], H[2,1,3,4], H[2,3,3,2] = t, t, t, t
    H[1,2,3,4], H[4,2,2,4], H[3,4,1,2], H[2,4,4,2] = t, t, t, t

    return H
end

function hamiltonian(model::FreeFermion)
    H = hamiltonian_array(model)
    V = ℤ₂Space(0=>2, 1=>2)
    TensorMap(H, V*V, V*V)
end