function getFieldValue(H, KG_slice_y, KG_slice_z, y, z)::ComplexF64

    return sum(H .* exp.(1im*KG_slice_y*y) .* exp.(1im*KG_slice_z*-z ))
end

function getD2field_poly2(k,c,lattice,z)
    if abs(c[1]) > abs(c[2])
        return (1+c[4]/c[1]*z/l.R+c[7]/c[1]*(z/l.R)^2)*exp(1im*k*z)
    else
        return (1+c[5]/c[2]*z/l.R+c[8]/c[2]*(z/l.R)^2)*exp(1im*k*z)
    end
end

function getD2field_poly4(k,c,lattice,z)
    znorm = l.R
    if abs(c[1]) > abs(c[2])
        return (1+c[4]/c[1]*(z/znorm)+c[7]/c[1]*(z/znorm)^2
        +c[10]/c[1]*(z/znorm)^3+c[13]/c[1]*(z/znorm)^4)*exp(1im*k*z)
    else
        return (1+c[5]/c[2]*z/znorm+c[8]/c[2]*(z/znorm)^2
        +c[11]/c[2]*(z/znorm)^3+c[14]/c[2]*(z/znorm)^4)*exp(1im*k*z)
    end
end

function getEfield_9D(k,c,lattice,params,z)
    k_v = [params.k_x ; params.k_y; k]
    H_inv = getHinv(Gs, k_v, params.k_1)
    Gzs = Gs[3,:,:,:]
    IP9D = [sinc.(V_2/2/pi * Gzs),
        - 1im ./(V_2 * Gzs) .* (sinc.(V_2/2/pi * Gzs) - cos.(V_2/2 * Gzs)),
        1/4 * sinc.(V_2/2/pi * Gzs) + 2 ./(V_2^2 * Gzs.^2) .* (cos.(V_2/2 * Gzs) - sinc.(V_2/2/pi * Gzs))]
    IP9D = [IP9D[n][i,j,k] for n in 1:3, i in 1:2*l.NG+1, j in 1:1, k in 1:1]
    IP9D[:,l.NG+1,:,:] = [1.0,0,1/12]
    multi = [kron(IP9D[:,k,l,m],H_inv[i,:,k,l,m]) for i in 1:3, k in 1:2*l.NG+1, l in 1:1, m in 1:1]
    multi = [multi[i,k,l,m][j] for i in 1:3, j in 1:9, k in 1:2*l.NG+1, l in 1:1, m in 1:1]
    @einsum εG[i,k,l,m] := multi[i,j,k,l,m]*c[j]
    phase = exp.(1im*(k .+ Gzs)*z)
    @einsum summand[k,l,m] := εG[2,k,l,m]*phase[k,l,m]
    return first(sum(summand,dims=(1,2,3)))
end


function getD2field_ana(lams,vs,params,lattice,z)
    a = lattice.A
    k2 = sqrt(params.e_m-sin(params.incl/180*pi)^2)*params.k_0
    return (vs[1,1]*exp(1im*k2*z) + vs[2,1]*exp(-1im*k2*z))/(vs[1,1]+vs[2,1])
end
