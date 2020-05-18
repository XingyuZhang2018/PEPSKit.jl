#=
    Can interpret vumps output in the pulling-through kind of way
    This gives 2 equivalent ways of obtaining corner matrices
        - lfps
        - rfps

    I then kinda take the average of the two, and use that
=#
function northwest_corner_tensors(init,nbound,npars,wbound,wpars,peps;verbose=true)
    #cornerprime
    (nrows,ncols) = size(peps)

    lfps = similar(init)
    for i in 1:nrows
        curl = [rightenv(wpars,s,nrows-i+1,wbound) for s in 1:ncols];
        botl = nbound.AL[i,1:ncols];

        (vals,vecs,convhist)=eigsolve(x->transfer_left(x,curl,botl),init[i,1],1,:LM,Arnoldi());
        convhist.converged == 0 && @info "lcorner failed to converge"
        lfps[i,1] = vecs[1]

        for j in 2:ncols
            lfps[i,j] = transfer_left(lfps[i,j-1],curl[j-1],botl[j-1])
        end
    end

    rfps = similar(init)
    for i in 1:ncols
        curr = [leftenv(npars,1-s,i,nbound) for s in 1:nrows];
        botr = wbound.AR[i,1:nrows];

        (vals,vecs,convhist)=eigsolve(x->transfer_right(x,curr,botr),init[1,i],1,:LM,Arnoldi());
        convhist.converged == 0 && @info "rcorner failed to converge"
        rfps[1,i] = vecs[1]

        for j in 2:nrows
            rfps[j,i] = transfer_right(rfps[j-1,i],curr[end-j+2],botr[end-j+2])
        end
    end

    toret = PeriodicArray(map(zip(lfps,rfps)) do (l,r)
        rmul!(r,dot(r,l));

        normalize!(l)
        normalize!(r)

        verbose && println("corner inconsistency $(norm(l-r))")

        0.5*(l+r)
    end)

    return toret
end