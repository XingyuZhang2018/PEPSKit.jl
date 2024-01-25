"""
    init_ipeps(model::HamiltonianModel; D::Int, χ::Int, tol::Real, maxiter::Int)
Initial `bcipeps` and give `key` for use of later optimization. The key include `model`, `D`, `χ`, `tol` and `maxiter`. 
The iPEPS is random initial if there isn't any calculation before, otherwise will be load from file `/data/model_D_chi_tol_maxiter.jld2`
"""
function init_ipeps(params::Params)
    @unpack model, iter, D, tol, verbose, infolder, contraction = params
    χ = contraction.χ
    Ni, Nj = model.Ni, model.Nj
    infolder = joinpath(infolder, "$(model)", "D$(D)_χ$(χ)")
    contraction.infolder = infolder
    contraction.outfolder = infolder
    params.infolder = infolder
    params.outfolder = infolder
    mkpath(infolder)
    chkp_file = joinpath(infolder, "ipeps_No$(iter).jld2")
    d = space(hamiltonian(model), 1)
    if isfile(chkp_file)
        ipeps = load(chkp_file)["ipeps"]
        verbose && println("load iPEPS from $chkp_file")
    else
        ipeps = TensorMap(randuniform, ComplexF64, D*D*d,D*D)
        verbose && println("random initial iPEPS $chkp_file")
    end
    ipeps /= norm(ipeps)
    return ipeps
end

function energy(ipeps, h, params::Params)
    @tensor Mp[-1 -2 -3 -4 9; -5 -6 -7 -8 10] := ipeps[-1 -3 9;-5 -7] * ipeps'[-6 -8;-2 -4 10]
    DD = fuse(params.D, params.D')
    d = space(ipeps, 3)
    Mp = tospace(Mp, DD * DD * d, DD * DD * d)
    @tensor M[-1 -2; -3 -4] := Mp[-1 -2 5; -3 -4 5]
    
    Env = obs_env(M, params.contraction)
    @unpack Eul, Eur, Edl, Edr, Elu, Eld, Elo, Eru, Erd, Ero = Env

    etol = 0.0

    """                                         
    1 ────┬──3 ─┬──── 5                        
    │     2     4     │                        
    ├─ 6 ─┼─ 7 ─┼─ 8 ─┤                       
    │     9     10    │                       
    11 ───┴─12 ─┴──── 13                       
    """                 
    @tensor lr[-14 -15; -16 -17] := Elo[1 6; 11] * Edl[11 9; 12] * Mp[6 2 -14; 9 7 -16] * Eul[3 2; 1] * Edr[12 10; 13] * Ero[13 8; 5] * Mp[7 4 -15; 10 8 -17] * Eur[5 4; 3]
    @plansor e = lr[-1 -2; -3 -4] * h[-3 -4; -1 -2]
    @plansor n = lr[-1 -2; -1 -2]
    etol += e / n

    """ 
    1 ────┬──── 3 
    │     2     │ 
    ├─ 4 ─┼─ 5 ─┤ 
    6     7     8 
    ├─ 9 ─┼─ 10─┤ 
    │     11    │ 
    12 ───┴─── 13 
    """ 
    @tensor ud[-14 -15; -16 -17] := Eru[8 5; 3] * Eul[3 2; 1] * Mp[4 2 -14; 7 5 -16] * Elu[1 4; 6] * Eld[6 9; 12] * Edl[12 11; 13] * Mp[9 7 -15; 11 10 -17] * Erd[13 10; 8]
    @plansor e = ud[-1 -2; -3 -4] * h[-3 -4; -1 -2]
    @plansor n = ud[-1 -2; -1 -2]
    etol += e / n

    return etol
end

_inner(x, dx1, dx2) = real(dot(dx1, dx2))

function optimise(ipeps, params::Params)
    h = hamiltonian(params.model)
    f(x) = real(energy(x, h, params))

    function fg(x)
        return f(x), gradient(f, x)[1]
    end
    # @show fg(ipeps)
    alg = LBFGS(; verbosity = 0, gradtol = params.tol, maxiter = params.maxiter)
    x, f, g, numfg, normgradhistory = optimize(fg, ipeps, alg; inner=_inner, callback=os->writelog(os, params))
    return x, f, g, numfg, normgradhistory
end

function writelog(os, params::Params)
    message = @sprintf("i = %5d,\tenergy = %.15f,\tgnorm = %.3e\n", os.iter, os.f, os.g)
    if params.verbose && os.iter % params.output_interval == 0
        printstyled(message; bold=true, color=:red)
        flush(stdout)
    end

    @unpack D, outfolder = params
    !(isdir(outfolder)) && mkdir(outfolder)

    if params.ifsave && os.iter % params.save_interval == 0
        logfile = open(joinpath(outfolder, "trace.log"), "a")
        write(logfile, message)
        close(logfile)
        save(joinpath(outfolder, "ipeps_No$(os.iter).jld2"), "ipeps", os.x)
    end
    return false
end