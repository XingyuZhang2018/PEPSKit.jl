using PEPSKit
using Random
using TensorKit
using ADFPCM: FPCM

let
    Random.seed!(10)
    params = PEPSKit.Params(model = FreeFermion(1,1), 
                            iff = true,
                            iter = 0, 
                            maxiter = 100,
                            D = ℤ₂Space(0=>2, 1=>2), 
                            save_interval = 1,
                            contraction = FPCM(χ=ℤ₂Space(0=>10, 1=>10), miniter=1, maxiter=100, verbose=true))
    ipeps = init_ipeps(params)
    optimise(ipeps, params)       
end
