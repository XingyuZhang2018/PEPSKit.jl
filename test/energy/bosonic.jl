using ADFPCM
using ADFPCM: num_grad
using PEPSKit
using PEPSKit: energy
using Random
using TensorKit
using Test
using Zygote

@testset "energy" begin
    Random.seed!(42)
    params = PEPSKit.Params(model = Heisenberg(1,1), 
                            D = ℂ^3, 
                            contraction = FPCM(χ=ℂ^8, miniter=10))

    ipeps = init_ipeps(params)
    h = hamiltonian(params.model)
    e = energy(ipeps, h, params, Val(:Bosonic))
    @test e ≈ 0.5 atol=1e-4
end

@testset "gradient" begin
    Random.seed!(42)
    params = PEPSKit.Params(model = Heisenberg(1,1), 
                            D=ℂ^2, 
                            contraction=FPCM(χ=ℂ^8, miniter=10, verbose=false))

    ipeps = init_ipeps(params)
    h = hamiltonian(params.model)
    f(x) = real(energy(x, h, params, Val(:Bosonic)))
    @test gradient(f, ipeps)[1] ≈ num_grad(f, ipeps) atol = 1e-4
end