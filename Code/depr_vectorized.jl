
function VecgetHinv(Gs, k_v, k_1)::Array{Complex{Float64},5}
    #Creating 𝓗⁻¹#
    # Adding vectors
    k_v = reshape(k_v, (3,1))
    @einsum kG[i,k,n,m] := k_v[i] + Gs[i,k,n,m]
    #Creating square of elements
    @einsum kG_2[k,n,m] := kG[i,k,n,m] * kG[i,k,n,m]
    #Creating TensorProduct
    @einsum outM[i,j,k,n,m] := kG[i,k,n,m] * kG[j,k,n,m]
    #Computing inverse
    H_factor = 1 ./ (k_1^2 .- kG_2)
    outM = Matrix{ComplexF64}(I,3,3) .- outM / k_1^2
    @einsum outM[i,j,k,n,m] = H_factor[k,n,m] * outM[i,j,k,n,m]
    return outM
end

function VecpolyxDiskIP(uuu,uuy,uuz,degmn)
    m=degmn[1]
    n=degmn[2]
    cαm = RecurCoef(degmn)
    Summands = zeros((floor(Int,m/2)+1,floor(Int,n/2)+1,2*l.NG+1,2*l.NG+1,1))
    for α in 0:floor(Int,m/2)
        for β in 0:floor(Int,n/2)
            Summands[α+1,β+1,:,:,:] = (-1)^(α+β) * cαm[m+1,α+1] * cαm[n+1,β+1] * uuy.^(m-2*α).*uuz.^(n-2*β)./uuu.^(m+n-α-β) .* BessQnoDC.(m+n-α-β+1,uuu)
        end
    end
    Summands[:,:,l.NG+1,l.NG+1,:] .= 0
    return dropdims(sum(Summands,dims=1:2),dims=(1,2))
end

function VecIPcoefficients(uuu,uuy,uuz,deg)
    Qqlen = (deg[1]+1)*(deg[2]+1)
    degreelist = zeros(Int8,(2,Qqlen))
    Qq = zeros((Qqlen,Qqlen))
    i=1
    while i <= Qqlen
        for n in 0:deg[2]
            for m in 0:deg[1]
                degreelist[:,i]=[m,n]
                i += 1
            end
        end
    end
    for i in 1:Qqlen
        for j in 1:Qqlen
            m=degreelist[1,i]+degreelist[1,j]
            n=degreelist[2,i]+degreelist[2,j]
            if isodd(m)
                Qq[i,j] = 0
            elseif isodd(n)
                Qq[i,j] = 0
            else
                Qq[i,j] = 2*doublefactorial(m-1)*doublefactorial(n-1)/ ( (m+n+2) * 2^((m+n)/2) * factorial((m+n)/2) )
            end
        end
    end

    IPvec = zeros(ComplexF64,(Qqlen,2*l.NG+1,2*l.NG+1,1))
    for i in 1:Qqlen
        degmn = degreelist[:,i]
        IPvec[i,:,:,:] = 2 * 1im^(degmn[1]+degmn[2]) * VecpolyxDiskIP(uuu,uuy,uuz,degmn)
    end

    Pp0 = Qq[1,:]*Qq[1,:]'

    return Qq,Pp0,IPvec
end

function VecgetpolyxM(deg, λ_value, eps = 1.0e-8)::Array{Complex{Float64},2}
    k_v = [p.k_x ; p.k_y; λ_value]
    H_inv = VecgetHinv(Gs, k_v, p.k_1)
    absGs = dropdims(sqrt.(sum(Gs.^2,dims=1)),dims=1)
    Gys = Gs[2,:,:,:]
    Gzs = Gs[3,:,:,:]
    uuu = absGs*l.R
    uuy = Gys*l.R
    uuz = Gzs*l.R
    Qq,Pp0,IPvec = VecIPcoefficients(uuu,uuy,uuz,deg)
    IPvec[:,l.NG+1,l.NG+1,1] = Qq[1,:]
    if deg[1]+deg[2] == 0
        IPvec = dropdims(IPvec,dims=1)
        IPvecconj = conj.(IPvec)
        Pp = IPvec.*IPvecconj
        summands = [Pp[k,n,m]*H_inv[:,:,k,n,m] for k in 1:2*l.NG+1, n in 1:2*l.NG+1, m in 1:1]
    else
        IPvecconj = conj.(IPvec)
        @einsum Pp[i,j,k,n,m] := IPvec[i,k,n,m] * IPvecconj[j,k,n,m]
        summands = [kron(Pp[:,:,k,n,m],H_inv[:,:,k,n,m]) for k in 1:2*l.NG+1, n in 1:2*l.NG+1, m in 1:1]
    end
    latsum = sum(summands)
    return kron(Qq,one(ones(3,3))) - ((p.k_1^2-p.k_2^2) * l.V_2 / l.V) * latsum
end

function VecgetpolyxMder(deg,λ_value, eps = 1.0e-8)::Complex{Float64}

    return (det(VecgetpolyxM(deg,λ_value+eps)) -
            det(VecgetpolyxM(deg,λ_value-eps)))/(2*eps)
end

function VecpolyxNewton(deg,init, maxiter=1000, tol2=5e-9)

    k = init
    for nn in 1:maxiter
        phik = det(VecgetpolyxM(deg,k))
        knew = k - phik / getpolyxMder(deg,k)
        delk = abs(1 - knew / k)
        if delk < tol2
            global knew = knew
            break
        end
        k = knew
        nn == maxiter ? throw("No conv in Newton") : nothing
    end
    return knew
end
