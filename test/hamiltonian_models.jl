using PEPSKit
using PEPSKit: HamiltonianModel
using TensorKit
using Test

@testset "Heisenberg" begin
    model = Heisenberg(1,1)
    @test model isa HamiltonianModel

    h = hamiltonian(model)
    @test h' == h
end

@testset "FreeFermion" begin
    model = FreeFermion(1,1)
    @test model isa HamiltonianModel

    h = hamiltonian(model)
    hh = reshape(h, 16,16)
    @test hh' == hh

    V = ℤ₂Space(0=>2, 1=>2)
    ht = TensorMap(h, V*V, V*V)
    @test norm(h) == norm(ht)
end