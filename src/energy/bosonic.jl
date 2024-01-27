function energy(ipeps, h, params::Params, ::Val{:Bosonic})
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
