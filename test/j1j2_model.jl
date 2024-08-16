using Test
using Random
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit

# initialize parameters
χbond = 2
χenv = 16
ctm_alg = CTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=2,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10)),
    ctmrgscheme=:simultaneous,
)
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; maxiter=100, gradtol=1e-3, verbosity=2),
    gradient_alg=LinSolver(; solver=GMRES(; tol=1e-6), iterscheme=:diffgauge),
    reuse_env=true,
)

# initialize states
Random.seed!(91283219347)
H = square_lattice_j1j2(; J2=0.25)
psi_init = InfinitePEPS(2, χbond)
psi_init = PEPSKit.symmetrize!(psi_init, PEPSKit.RotateReflect())
env_init = leading_boundary(CTMRGEnv(psi_init, ComplexSpace(χenv)), psi_init, ctm_alg);

# find fixedpoint
finalize = (args...) -> PEPSKit.symmetrize_finalize(args..., RotateReflect())
result = fixedpoint(psi_init, H, opt_alg, env_init; finalize)
ξ_h, ξ_v, = correlation_length(result.peps, result.env)

# compare against Juraj Hasik's data:
# https://github.com/jurajHasik/j1j2_ipeps_states/blob/main/single-site_pg-C4v-A1/j20.25/state_1s_A1_j20.25_D2_chi_opt48.dat
ξ_ref = -1 / log(0.2723596743547324)
@test result.E ≈ -0.5618837021945925 atol = 1e-3
@test all(@. isapprox(ξ_h, ξ_ref; atol=1e-1) && isapprox(ξ_v, ξ_ref; atol=1e-1))
