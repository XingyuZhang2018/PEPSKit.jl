using ADFPCM
using PEPSKit
using PEPSKit: energy
using TensorKit
using Test

@testset "init_ipeps" begin
    params = PEPSKit.Params(model=Heisenberg(1,1), D=ℂ^3, contraction=FPCM(χ=ℂ^8))
    ipeps = init_ipeps(params)
    @test space(ipeps) == (ℂ^3 * ℂ^3 * ℂ^2 ← ℂ^3 * ℂ^3)
end

@testset "energy" begin
    params = PEPSKit.Params(model=Heisenberg(1,1), D=ℂ^3, contraction=FPCM(χ=ℂ^8, miniter=10))
    ipeps = init_ipeps(params)
    e = energy(ipeps, params)
    @show e
end