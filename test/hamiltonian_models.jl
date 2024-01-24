using PEPSKit
using PEPSKit: HamiltonianModel
using Test

@testset "hamiltonian" begin
    model = Heisenberg(1,1)
    @test model isa HamiltonianModel

    h = hamiltonian(model)
    @test h' == h
end