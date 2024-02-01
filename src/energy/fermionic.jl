function swapgate(V1::VectorSpace, V2::VectorSpace)
    I1 = isomorphism(V1, V1)
    I2 = isomorphism(V2, V2)
    S = I1 ⊗ I2
    ind = V1.dims[1]*V2.dims[1] + 1
    S.data[Z2Irrep(0)][ind:end, ind:end] .*= -1 # current only for Z2 symmetry 
    return S
end

@non_differentiable swapgate(V1, V2)

"""
    fdag(ipeps, SDD)

Obtain dag tensor for local peps tensor in Fermionic Tensor Network(by inserting swapgates). The input tensor has indices which labeled by (lurdf)
legs are counting from f and clockwisely.

input legs order: ulfdr
output legs order: ulfdr
"""
fdag(ipeps, SDD) = @plansor ipepsdag[-6 -7; -9 -8 -3] := SDD[-6 -7; 4 5] * ipeps'[4 5; 1 2 -3] * SDD[1 2; -9 -8]

"""
    double_layer(ipeps, SDD)
    
Obtain M tensor in ipeps, while the input tensor has indices which labeled by (lurdf).
This tensor is ready for contraction algorithm
"""
function double_layer(ipeps, SDD)
    ipepsdag = fdag(ipeps, SDD)
    @tensor M[-7 -12 -13 -1; -8 -10 -11 -5] := ipepsdag[-8 9; 6 -7 3] * ipeps[-1 2 3; 4 -5] * SDD[4 -11; -10 9] * SDD[-12 6; 2 -13]

    D = space(ipeps, 1)
    DD = Zygote.@ignore fuse(D', D)
    It = Zygote.@ignore isomorphism(DD, D'*D)

    @tensor M[-1 -2; -3 -4] := It[-1; 1 2] * It[-2; 3 4] * M[1 2 3 4; 5 6 7 8] * It'[5 6; -3] * It'[7 8; -4]
    return M
end

"""
    energy(ipeps, h, params, ::Val{:Bosonic})
"""
function energy(ipeps, h, params::Params, ::Val{:Fermionic})
    D = params.D
    DD = Zygote.@ignore fuse(D', D)
    d = space(ipeps, 3)
    SDD = swapgate(D, D)
    SdD = swapgate(d, D)
    ipepsdag = fdag(ipeps, SDD)

    M = double_layer(ipeps, SDD)
    Env = obs_env(M, params.contraction)
    # envir = env(M, params.contraction)
    # Cul = convert(Array, envir.Cul)
    # @show diag(Cul)
    @unpack Eul, Eur, Edl, Edr, Elu, Eld, Elo, Eru, Erd, Ero = Env

    etol = 0.0

    It = Zygote.@ignore isomorphism(D'*D, DD)
    Eul, Eur, Elu, Eld, Elo = map(x->(@tensor y[-1 -2 -3; -4] := It'[1; -2 -3] * x[-1 1; -4]), (Eul, Eur, Elu, Eld, Elo))
    Edl, Edr, Eru, Erd, Ero = map(x->(@tensor y[-1 -2 -3; -4] := It[-2 -3; 1] * x[-1 1; -4]), (Edl, Edr, Eru, Erd, Ero))

    etol = 0.0

    # To do: finish the following code 
    @tensoropt lr[-21 -24; -23 -25] := Elo[39 7 29; 38] * Edl[38 9 40; 37] * ipepsdag[9 10; 6 7 8] * SDD[17 27; 10 40] * SdD[8 22; -23 27] * Eul[34 30 1; 39] * ipeps[1 2 3; 4 5] * SDD[29 6; 2 30] * SdD[-21 4; 3 22] * Edr[37 19 32; 36] * ipepsdag[19 20; 16 17 18] * SDD[14 31; 32 20] * SdD[18 16; -25 26] * Eur[35 33 11; 34] * ipeps[11 12 13; 14 15] * SDD[5 28; 12 33] * SdD[-24 26; 13 28] * Ero[36 31 15; 35] 
    @plansor e = lr[1 2; 3 4] * h[3 4; 1 2]
    @plansor n = lr[1 2; 1 2]
    abs(n) < 1e-8 && println("overlap is zero. e = $e, n = $n")
    etol += e / n

    @tensoropt ud[-15 -30; -14 -32] := Eul[36 12 1; 35] * Eru[37 18 5; 36] * ipeps[1 2 3; 4 5] * SDD[6 11; 12 2] * SdD[-15 16; 3 13] * Elu[35 7 11; 40] * ipepsdag[9 10; 6 7 8] * SDD[23 34; 9 33] * SdD[8 13; -14 10] * Eld[40 24 34; 39] * ipepsdag[26 27; 23 24 25] * SDD[29 21; 27 28] * SdD[25 33; -32 31] * Edl[39 26 28; 38] * ipeps[17 19 20; 21 22] * SDD[4 18; 17 16] * SdD[-30 31; 20 19] * Erd[38 29 22; 37]
    @plansor e = ud[1 2; 3 4] * h[3 4; 1 2]
    @plansor n = ud[1 2; 1 2]
    abs(n) < 1e-8 && println("overlap is zero. e = $e, n = $n")
    etol += e / n

    @show etol
    return etol
end
