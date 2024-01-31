using ADFPCM: Algorithm

@with_kw mutable struct Params
    model::HamiltonianModel
    iff::Bool = false # if fermionic tensor network
    iter::Int = 0 # load from No.iter ipeps 
    D::VectorSpace # bond dimension
    tol::Float64 = 1e-6 # optimise convergence tolerance
    maxiter::Int = 100 # optimise max iteration
    output_interval::Int = 1 # output interval
    ifsave::Bool = true # if save intermediate results
    savetol::Float64 = 1e-1 # save tolerance
    save_interval::Int = 10 # save interval
    infolder = "./data/" # load from infolder
    outfolder = infolder # save to outfolder
    verbose::Bool = true # if print log
    contraction::Algorithm # contraction algorithm
    backmaxiter::Int = 10 # backward max iteration
    backminiter::Int = 3 # backward min iteration
end
