using PEPSKit
using Random

let
    Random.seed!(10)
    params = PEPSKit.Params(model = FreeFermion(1,1), 
                            iff = true,
                            iter = 4, 
                            maxiter = 20,
                            D = ℤ₂Space(0=>2, 1=>1), 
                            save_interval = 1,
                            contraction = FPCM(χ=ℤ₂Space(0=>10, 1=>10), miniter=1, maxiter=20, verbose=false))
    ipeps = init_ipeps(params)
    optimise(ipeps, params)       
end
