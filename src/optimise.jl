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

function optimise(ipeps, params::Params)
    h = hamiltonian(params.model)
    f(x) = params.iff ? real(energy(x, h, params, Val(:Fermionic))) : real(energy(x, h, params, Val(:Bosonic)))

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
        logfile = open(rejoinpath(outfolder, "trace.log"), "a")
        write(logfile, message)
        close(logfile)
        save(rejoinpath(outfolder, "ipeps_No$(os.iter).jld2"), "ipeps", os.x)
    end
    return false
end