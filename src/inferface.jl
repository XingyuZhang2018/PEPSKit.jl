using ADFPCM: Algorithm

@with_kw mutable struct Params
    model::HamiltonianModel
    D::VectorSpace
    tol::Float64 = 1e-14
    optmaxiter::Int = 1000
    optminiter::Int = 100
    output_interval::Int = 1
    ifsave::Bool = true
    savetol::Float64 = 1e-1
    save_interval::Int = 10
    infolder = "./data/"
    outfolder = infolder
    verbose::Bool = true
    contraction::Algorithm
end
