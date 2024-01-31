using ADFPCM: rejoinpath

"""
    init_ipeps(params::Params)

The iPEPS is random initial if there isn't any calculation before, otherwise will be load from file `/data/model_D_chi_tol_maxiter.jld2`
"""
function init_ipeps(params::Params)
    @unpack model, iter, D, tol, verbose, infolder, contraction = params
    χ = contraction.χ

    infolder = rejoinpath(infolder, "$(model)", "D$(D)_χ$(χ)")
    contraction.infolder = infolder
    contraction.outfolder = infolder
    params.infolder = infolder
    params.outfolder = infolder
    mkpath(infolder)
    chkp_file = rejoinpath(infolder, "ipeps_No$(iter).jld2")
    if isfile(chkp_file)
        ipeps = load(chkp_file)["ipeps"]
        verbose && println("load iPEPS from $chkp_file")
    else
        d = space(hamiltonian(model), 1)
        ipeps = TensorMap(randuniform, ComplexF64, D*D*d,D*D)
        verbose && println("random initial iPEPS $chkp_file")
    end

    ipeps /= norm(ipeps)
    return ipeps
end

_inner(x, dx1, dx2) = real(dot(dx1, dx2))
function _finalize!(x, f, g, iter, params)
    message = @sprintf("i = %5d,\tenergy = %.15f,\tgnorm = %.3e\n", iter, f, norm(g))
    if params.verbose && iter % params.output_interval == 0
        printstyled(message; bold=true, color=:red)
        flush(stdout)
    end

    @unpack D, outfolder = params
    !(isdir(outfolder)) && mkdir(outfolder)

    if params.ifsave && iter % params.save_interval == 0
        logfile = open(rejoinpath(outfolder, "trace.log"), "a")
        write(logfile, message)
        close(logfile)
        save(rejoinpath(outfolder, "ipeps_No$(iter).jld2"), "ipeps", x)
    end
    return x, f, g
end 

function optimise(ipeps, params::Params)
    h = hamiltonian(params.model)
    paramsback = deepcopy(params)
    paramsback.contraction.maxiter = params.backmaxiter
    paramsback.contraction.miniter = params.backminiter
    f(x) = params.iff ? real(energy(x, h, params, Val(:Fermionic))) : real(energy(x, h, params, Val(:Bosonic)))
    fback(x) = params.iff ? real(energy(x, h, paramsback, Val(:Fermionic))) : real(energy(x, h, paramsback, Val(:Bosonic)))

    function fg(x)
        return f(x), gradient(fback, x)[1]
    end
    # @show fg(ipeps)
    alg = LBFGS(; verbosity = 0, gradtol = params.tol, maxiter = params.maxiter)
    x, f, g, numfg, normgradhistory = optimize(fg, ipeps, alg; inner = _inner, finalize! = (x, f, g, iter)->_finalize!(x, f, g, iter, params))
    return x, f, g, numfg, normgradhistory
end