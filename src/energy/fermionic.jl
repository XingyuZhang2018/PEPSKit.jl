function swapgate(V1::VectorSpace, V2::VectorSpace)
    I1 = isomorphism(V1, V1)
    I2 = isomorphism(V2, V2)
    S = I1 ⊗ I2
    ind = V1.dims[1]*V2.dims[1] + 1
    S.data[Z2Irrep(0)][ind:end, ind:end] .*= -1
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
    @tensor M[-8 -12 -1 -11;-13 -5 -7 -10] := SDD[-12 6; 2 -13] * ipeps[-1 2 3; 4 -5] * ipepsdag[-8 9; 6 -7 3] * SDD[4 -11; -10 9]
    M = permute(M, ((7,2,5,3),(1,8,4,6)))
    D = space(ipeps, 1)
    DD = fuse(D', D)
    tospace(M, DD * DD, DD * DD)
end

"""
    energy(ipeps, h, params, ::Val{:Bosonic})
"""
function energy(ipeps, h, params::Params, ::Val{:Fermionic})
    M = double_layer(ipeps, swapgate(params.D, params.D))
    
    Env = obs_env(M, params.contraction)
    # envir = env(M, params.contraction)
    # Cul = envir.Cul
    # @show Cul
    @unpack Eul, Eur, Edl, Edr, Elu, Eld, Elo, Eru, Erd, Ero = Env

    etol = 0.0

    # To do: finish the following code
    # """                                         
    # 1 ────┬──3 ─┬──── 5                        
    # │     2     4     │                        
    # ├─ 6 ─┼─ 7 ─┼─ 8 ─┤                       
    # │     9     10    │                       
    # 11 ───┴─12 ─┴──── 13                       
    # """                 
    # @tensor lr[-14 -15; -16 -17] := Elo[1 6; 11] * Edl[11 9; 12] * Mp[6 2 -14; 9 7 -16] * Eul[3 2; 1] * Edr[12 10; 13] * Ero[13 8; 5] * Mp[7 4 -15; 10 8 -17] * Eur[5 4; 3]
    # @plansor e = lr[-1 -2; -3 -4] * h[-3 -4; -1 -2]
    # @plansor n = lr[-1 -2; -1 -2]
    # etol += e / n

    # """ 
    # 1 ────┬──── 3 
    # │     2     │ 
    # ├─ 4 ─┼─ 5 ─┤ 
    # 6     7     8 
    # ├─ 9 ─┼─ 10─┤ 
    # │     11    │ 
    # 12 ───┴─── 13 
    # """ 
    # @tensor ud[-14 -15; -16 -17] := Eru[8 5; 3] * Eul[3 2; 1] * Mp[4 2 -14; 7 5 -16] * Elu[1 4; 6] * Eld[6 9; 12] * Edl[12 11; 13] * Mp[9 7 -15; 11 10 -17] * Erd[13 10; 8]
    # @plansor e = ud[-1 -2; -3 -4] * h[-3 -4; -1 -2]
    # @plansor n = ud[-1 -2; -1 -2]
    # etol += e / n

    return etol
end
