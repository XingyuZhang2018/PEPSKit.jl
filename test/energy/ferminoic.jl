using ADFPCM
using ADFPCM: num_grad
using PEPSKit
using PEPSKit: energy, swapgate, fdag, double_layer
using Random
using TensorKit
using Test
using Zygote

@testset "swapgate" begin
    V1 = ℤ₂Space(0=>1, 1=>2)
    V2 = ℤ₂Space(0=>3, 1=>4)
    S = swapgate(V1, V2)
    @test S.data[Z2Irrep(0)][4:11, 4:11] == -I
end

@testset "fdag double_layer" begin
    D = ℤ₂Space(0=>2, 1=>1)
    d = ℤ₂Space(0=>2, 1=>2)
    S = swapgate(D, D)
    params = PEPSKit.Params(model=FreeFermion(1,1), D=D, contraction=FPCM(χ=D), verbose=false)
    ipeps = init_ipeps(params)
    SDD = swapgate(D, D)

    ipepsdag = fdag(ipeps, SDD)
    @test space(ipepsdag) == (D * D ← D * D * d)
    M = double_layer(ipeps, SDD)
    DD = fuse(D', D)
    @test space(M) == (DD * DD ← DD * DD)
end

@testset "energy" begin
    Random.seed!(42)
    params = PEPSKit.Params(model = FreeFermion(1,1), 
                            iff = true,
                            ifsave=false,
                            D = ℤ₂Space(0=>1, 1=>1), 
                            contraction = FPCM(χ=ℤ₂Space(0=>10, 1=>10), miniter=10))

    ipeps = init_ipeps(params)
    h = hamiltonian(params.model)
    e = energy(ipeps, h, params, Val(:Fermionic))
    @test e ≈ -0.02488097 atol=1e-4
end