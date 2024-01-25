using ADFPCM
using ADFPCM: num_grad
using PEPSKit
using PEPSKit: energy
using TensorKit
using Test
using Zygote

@testset "init_ipeps" begin
    params = PEPSKit.Params(model=Heisenberg(1,1), D=ℂ^3, contraction=FPCM(χ=ℂ^8))
    ipeps = init_ipeps(params)
    @test space(ipeps) == (ℂ^3 * ℂ^3 * ℂ^2 ← ℂ^3 * ℂ^3)
end

@testset "energy" begin
    Random.seed!(42)
    params = PEPSKit.Params(model=Heisenberg(1,1), D=ℂ^2, contraction=FPCM(χ=ℂ^8, miniter=10))
    ipeps = init_ipeps(params)
    h = hamiltonian(params.model)
    e = energy(ipeps, h, params)
    @test e ≈ 0.5 atol=1e-4
end

@testset "gradient" begin
    Random.seed!(42)
    params = PEPSKit.Params(model=Heisenberg(1,1), D=ℂ^2, contraction=FPCM(χ=ℂ^8, miniter=10, verbose=false))
    ipeps = init_ipeps(params)
    h = hamiltonian(params.model)
    f(x) = real(energy(x, h, params))
    @test gradient(f, ipeps)[1] ≈ num_grad(f, ipeps) atol = 1e-4
end

@testset "gradient" begin
    Random.seed!(10)
    params = PEPSKit.Params(model=Heisenberg(1,1,-1.0,-1.0,1.0), 
                            D=ℂ^2, 
                            save_interval=10,
                            contraction=FPCM(χ=ℂ^10, miniter=1, maxiter=20, verbose=false))

    ipeps = init_ipeps(params)
    @test optimise(ipeps, params)[2] ≈ -0.662513783652071 atol = 1e-6
end