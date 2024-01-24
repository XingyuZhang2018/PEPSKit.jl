using Test
using OptimKit
using LinearAlgebra

@testset "Optimization Algorithm " begin
    function quadraticproblem(B, y)
        function fg(x)
            g = B*(x-y)
            f = dot(x-y, g)/2
            return f, g
        end
        return fg
    end

    n = 2
    y = randn(n)
    A = randn(n, n)
    fg = quadraticproblem(A'*A, y)
    x₀ = randn(n)
    alg = LBFGS(; verbosity = 1, gradtol = 1e-12)
    x, f, g, numfg, normgradhistory = optimize(fg, x₀, alg)
    @test x ≈ y
    @test f < 1e-14

    @show x f g numfg normgradhistory
end