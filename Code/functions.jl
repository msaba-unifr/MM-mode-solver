
function InnerProd(x,mmdim;exclude_DC=false)
    #
    if mmdim == 2
        if x == 0
            if exclude_DC
                return 0
            else
                return l.V_2  # 2 * (1/2) * l.V_2
            end
        end
        return 2 * l.V_2 / x * besselj(1,x)
    elseif mmdim == 1
        if x == 0
            if exclude_DC
                return 0
            else
                return l.V_2
            end
        end
        return l.V_2 * sin(x)/x
    end
end

function BessQnoDC(n,x)
    #
    if x == 0
        return 0
    else
        return besselj(n,x)/x
    end
end


function getGspace(mmdim)
    space_ax = -l.NG:1.0:l.NG
    if mmdim == 1
        Gs = [l.B*[h,k,n] for h in space_ax, k in 0:0, n in 0:0]
        Gs = [Gs[h,k,n][i] for i in 1:3, h in 1:2*l.NG+1, k in 1:1, n in 1:1]
    elseif mmdim == 2
        Gs = [l.B*[h,k,n] for h in space_ax, k in space_ax, n in 0:0]
        Gs = [Gs[h,k,n][i] for i in 1:3, h in 1:2*l.NG+1, k in 1:2*l.NG+1, n in 1:1]
    end
    return Gs
end

function getHinv(G, k_v, k_1)::Array{Complex{Float64},2}
    #Creating 𝓗⁻¹#
    # Adding vectors
    k_v = reshape(k_v, (3,1))
    kG = k_v + G
    #Creating TensorProduct
    outM = kG * kG'
    #Computing inverse
    H_factor = 1 ./ (k_1^2 .- sum(kG.^2))
    outM = Matrix{ComplexF64}(I,3,3) .- outM / k_1^2
    outM = H_factor .* outM
    return outM
end

function getInitGuess(InnerP, H_inv, k_1, k_2, k_x, k_y, V_2, V)
    #InitialGuess
    ζ = (k_1^2-k_2^2) * V_2 / V / k_1^2
    @einsum Mm[i,j] :=  InnerP[k,n,m] * H_inv[i,j,k,n,m]
    Mm = I - k_1^2 / V_2^2 * ζ * Mm
    A2 = Mm - ζ * [0.0 0 0; 0 0 0; 0 0 1]
    A1 = -ζ * (k_x *[0.0 0 1; 0 0 0; 1 0 0] + k_y *[0.0 0 0; 0 0 1; 0 1 0])
    A0 = ζ * (k_1^2 * I - k_x^2 *[1.0 0 0; 0 0 0; 0 0 0] -
        k_y^2 *[0.0 0 0; 0 1 0; 0 0 0] - k_x * k_y *
        [0.0 1 0; 1 0 0; 0 0 0]) - (k_1^2 - k_x^2 - k_y^2) * Mm
    QEVP_LH = [A1 A0; -I zeros((3,3))]
    QEVP_RH = -[A2 zeros((3,3)); zeros((3,3)) I]
    return eigen(QEVP_LH,QEVP_RH)
end

function polyxDiskIP(G ,degmn)
    m=degmn[1]
    n=degmn[2]
    abs_G = sum(G.^2)
    if abs_G == 0
        return 0.0 + 0.0im
    end
    uuu = abs_G*l.R
    Summands = zeros((floor(Int,m/2)+1,floor(Int,n/2)+1))
    uuuy = G[2]*l.R
    uuuz = G[3]*l.R
    cαm = RecurCoef(degmn)
    for α in 0:floor(Int,m/2)
        for β in 0:floor(Int,n/2)
            Summands[α+1,β+1] = (-1)^(α+β) * cαm[m+1,α+1] * cαm[n+1,β+1] *
                uuuy^(m-2*α)*uuuz^(n-2*β)/uuu^(m+n-α-β) *
                    BessQnoDC(m+n-α-β+1,uuu)
                    if G == zeros(size(G))
                        println("G= ", G, "  ||  ", Summands[α+1,β+1])
                    end
        end
    end
    #Summands[:,:,l.NG+1,l.NG+1,:] .= 0 #should not be necessary
    return sum(Summands)
end

function RecurCoef(degmn)
    cαm = zeros((maximum(degmn)+1,floor(Int,maximum(degmn)/2)+1))
    cαm[1,:].=0
    cαm[:,1].=1
    for i in 2:maximum(degmn)+1
        for j in 2:floor(Int,maximum(degmn)/2)+1
            cαm[i,j] = cαm[i-1,j] + (i-2-2*(j-2))*cαm[i-1,j-1]
        end
    end
    return cαm
end

function getQq(deg)
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

    Pp0 = Qq[1,:]*Qq[1,:]'
    return Qq, Pp0, degreelist
end

function getIPvec(G, deg, degreelist)

    IPvec = zeros(ComplexF64,(size(degreelist,2)))
    for i in 1:((deg[1]+1)*(deg[2]+1))
        degmn = degreelist[:,i]
        IPvec[i] = 2 * 1im^(degmn[1]+degmn[2]) * polyxDiskIP(G ,degmn)
    end
    return IPvec
end

function getQEPpolyx(deg, H_inv, k_1, k_2, k_x, k_y, V_2, V)
    ζ = (k_1^2-k_2^2) * V_2 / V / k_1^2
    absGs = dropdims(sqrt.(sum(Gs.^2,dims=1)),dims=1)
    Gys = Gs[2,:,:,:]
    Gzs = Gs[3,:,:,:]
    uuu = absGs*l.R
    uuy = Gys*l.R
    uuz = Gzs*l.R
    Qq,Pp0,IPvec = IPcoefficients(uuu,uuy,uuz,deg)
    if deg[1]+deg[2] == 0
        IPvec = dropdims(IPvec,dims=1)
        IPvecconj = conj.(IPvec)
        Pp = IPvec.*IPvecconj
        Kk = [Pp[k,n,m]*H_inv[:,:,k,n,m] for k in 1:2*l.NG+1, n in 1:2*l.NG+1, m in 1:1]
    else
        IPvecconj = conj.(IPvec)
        @einsum Pp[i,j,k,n,m] := IPvec[i,k,n,m] * IPvecconj[j,k,n,m]
        Kk = [kron(Pp[:,:,k,n,m],H_inv[:,:,k,n,m]) for k in 1:2*l.NG+1, n in 1:2*l.NG+1, m in 1:1]
    end
    Kk = sum(Kk)
    Kk = kron(Qq,one(ones(3,3))) - (k_1^2-k_2^2)* V_2/V * Kk
    A2 = Kk - ζ * kron(Pp0,[0.0 0 0; 0 0 0; 0 0 1])
    A1 = -ζ * kron(Pp0,(k_x *[0.0 0 1; 0 0 0; 1 0 0] + k_y *[0.0 0 0; 0 0 1; 0 1 0]))
    A0 = ζ * kron(Pp0,(k_1^2 * I - k_x^2 *[1.0 0 0; 0 0 0; 0 0 0] -
        k_y^2 *[0.0 0 0; 0 1 0; 0 0 0] - k_x * k_y *
        [0.0 1 0; 1 0 0; 0 0 0])) - (k_1^2 - k_x^2 - k_y^2) * Kk
    QEVP_LH = [A1 A0; -I zeros(size(A2))]
    QEVP_RH = -[A2 zeros(size(A2)); zeros(size(A2)) I]
    return eigen(QEVP_LH,QEVP_RH)
end

function getpolyxM(deg, λ_value, NG, lat, eps = 1.0e-8)::Array{Complex{Float64},2}

    k_v = [p.k_x ; p.k_y; λ_value]
    Qq, Pp0, deg_list = getQq(deg)
    summands = zeros(ComplexF64, (3*(deg[1]+1)*(deg[2]+1),3*(deg[1]+1)*(deg[2]+1)))
    for k in -NG:NG, n in -NG:NG, m in 1:1

        G = lat * [k, n, m]
        H_inv = getHinv(G, k_v, p.k_1)
        if k == NG+1 && n == NG+1
            IPvec = Qq[1,:]
        else
            IPvec = getIPvec(G, deg, deg_list)
        end
        summands += kron((IPvec * conj(IPvec)'), H_inv)
        #println("k= ", k, "n= ", n , "  ||  ", maximum(abs.(kron((IPvec * conj(IPvec)'), H_inv))))
    end
    return kron(Qq,one(ones(3,3))) - ((p.k_1^2-p.k_2^2) * l.V_2 / l.V) * summands
end

function getpolyxMder(deg,λ_value, eps = 1.0e-8)::Complex{Float64}

    return (det(getpolyxM(deg,λ_value+eps)) -
            det(getpolyxM(deg,λ_value-eps)))/(2*eps)
end

function polyxNewton(deg,init, maxiter=1000, tol2=5e-9)

    k = init
    for nn in 1:maxiter
        phik = det(getpolyxM(deg,k))
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

function doublefactorial(n::Integer)
    if n <= 0
        return 1
    elseif isodd(n)
        k = (n+1)/2
        return factorial(2*k) / 2^k / factorial(k)
    else
        k = n/2
        return 2^k * factorial(k)
    end
end

function solve_analytical(params,lattice,TE=true)
    eps1 = params.e_bg; eps2 = params.e_m
    println(); println("eps_m = ",eps2)
    k1 = sqrt(eps1-sin(params.incl/180*pi)^2)*params.k_0
    k2 = sqrt(eps2-sin(params.incl/180*pi)^2)*params.k_0
    d2 = lattice.V_2; d1 = lattice.A-d2
    Z1 = k1; Z2 = k2
    if TE==false
        Z1 = k1/eps1; Z2 = k2/eps2
    end
    Up = (cos(k1*d1)+1im/2*(Z1/Z2+Z2/Z1)*sin(k1*d1))*exp( 1im*k2*d2)
    Um = (cos(k1*d1)-1im/2*(Z1/Z2+Z2/Z1)*sin(k1*d1))*exp(-1im*k2*d2)
    V = 1im/2*(Z2/Z1-Z1/Z2)*sin(k1*d1)
    T = [Up V; -V Um]
    return eigen(T)
end
