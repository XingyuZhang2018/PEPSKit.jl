using ADFPCM
using ADFPCM: num_grad
using PEPSKit
using TensorKit
using Test
using Zygote

@testset "init_ipeps" begin
    params = PEPSKit.Params(model=Heisenberg(1,1), D=ℂ^3, contraction=FPCM(χ=ℂ^8))
    ipeps = init_ipeps(params)
    @test space(ipeps) == (ℂ^3 * ℂ^3 * ℂ^2 ← ℂ^3 * ℂ^3)
end

@testset "Heisenberg Bosonic" begin
    Random.seed!(10)
    params = PEPSKit.Params(model = Heisenberg(1,1,-1.0,-1.0,1.0), 
                            iff = false,
                            D = ℂ^2, 
                            save_interval = 10,
                            contraction = FPCM(χ=ℂ^10, miniter=1, maxiter=20, verbose=false))

    ipeps = init_ipeps(params)
    @test optimise(ipeps, params)[2] ≈ -0.662513783652071 atol = 1e-6
end

@testset "FreeFermion Fermionic" begin
    Random.seed!(10)
    params = PEPSKit.Params(model = FreeFermion(1,1), 
                            iff = true,
                            iter = 0, 
                            maxiter = 20,
                            D = ℤ₂Space(0=>1, 1=>1), 
                            save_interval = 1,
                            contraction = FPCM(χ=ℤ₂Space(0=>5, 1=>5), miniter=1, maxiter=20, verbose=false))

    ipeps = init_ipeps(params)
    @test optimise(ipeps, params)[2] ≈ -1.45 atol = 1e-1
end