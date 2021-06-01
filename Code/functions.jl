struct Parameters
    lambda::AbstractFloat
    azim::AbstractFloat
    incl::AbstractFloat
    e_m::Complex
    e_bg::AbstractFloat
    k_0::AbstractFloat
    k_x::AbstractFloat
    k_y::AbstractFloat
    k_1::Complex{Float64}
    k_2::Complex
end

Parameters(lambda, azim, incl, e_m, e_bg) = Parameters(lambda,azim,incl,e_m,e_bg,
    2*pi/lambda,
    2*pi/lambda*cos(azim/180*pi) * sin(incl/180*pi),
    2*pi/lambda*sin(azim/180*pi) * sin(incl/180*pi),
    2*pi/lambda*sqrt(e_bg),
    2*pi/lambda*sqrt(e_m))

struct Lattice2D
    NG
    A
    V_2
    R
    B
    V
end

struct Lattice3D
    NG
    A
    V_2
    R
    B
    V
end

struct Lattice1D
    NG
    A
    V_2
    R
    B
    V
end


Lattice3D(NG,A,V_2,R) = Lattice3D(NG, A, V_2, R, 2*pi*inv(A)', abs(det(A)))

#Lattice2D(NG,A,V_2) = Lattice2D(NG, A, V_2,
#2*pi*[0 0 0; inv(A)[1,1] inv(A)[2,1] 0; inv(A)[1,2] inv(A)[2,2] 0],
#det(A))

Lattice2D(NG,A,V_2,R) = Lattice2D(NG, A, V_2, R,
2*pi*[0 0 0; inv(A)[1,1] inv(A)[2,1] 0; inv(A)[1,2] inv(A)[2,2] 0],
abs(det(A)))

Lattice1D(NG,a,V_2,R) = Lattice1D(NG, a, V_2, R,
2*pi/a*[0 0 0; 0 0 0; 1 0 0],
a)

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

function getHinv(Gs, k_v, k_1)::Array{Complex{Float64},5}
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

function getQEPpoly2(H_inv, k_1, k_2, k_x, k_y, V_2, V)
    #InitialGuess
    ζ = (k_1^2-k_2^2) * V_2 / V / k_1^2
    absGs = dropdims(sqrt.(sum(Gs.^2,dims=1)),dims=1)
    # Gys = Gs[2,:,:,:]
    Gzs = Gs[3,:,:,:]
    uuu = Gzs*l.R
    IPvec = [sin.(uuu)./uuu,
             1im ./ uuu.^2 .* (sin.(uuu) - uuu.*cos.(uuu)),
             1   ./ uuu.^3 .* (2*uuu.*cos.(uuu) + (uuu.^2 .- 2).*sin.(uuu))]
    IPvec = [IPvec[n][i,j,k] for n in 1:3, i in 1:2*l.NG+1, j in 1:1, k in 1:1]
    IPvec[:,l.NG+1,:,:] .= 0
    IPvecconj = conj.(IPvec)
    @einsum Pp[i,j,k,n,m] := IPvec[i,k,n,m] * IPvecconj[j,k,n,m]
    Kk = [kron(Pp[:,:,k,n,m],H_inv[:,:,k,n,m]) for k in 1:2*l.NG+1, n in 1:1, m in 1:1]
    Kk = sum(Kk)
    Qq = [1.0+0im 0 1/3;0 1/3 0;1/3 0 1/5]
    Pp0 = [1.0+0im 0 1/3;0 0 0;1/3 0 1/9]
    Kk = kron(Qq,one(ones(3,3))) - (k_1^2-k_2^2)* V_2/V * Kk
    A2 = Kk - ζ * kron(Pp0,[0.0 0 0; 0 0 0; 0 0 1])
    A1 = -ζ * kron(Pp0,(k_x *[0.0 0 1; 0 0 0; 1 0 0] + k_y *[0.0 0 0; 0 0 1; 0 1 0]))
    A0 = ζ * kron(Pp0,(k_1^2 * I - k_x^2 *[1.0 0 0; 0 0 0; 0 0 0] -
        k_y^2 *[0.0 0 0; 0 1 0; 0 0 0] - k_x * k_y *
        [0.0 1 0; 1 0 0; 0 0 0])) - (k_1^2 - k_x^2 - k_y^2) * Kk
    QEVP_LH = [A1 A0; -I zeros((9,9))]
    QEVP_RH = -[A2 zeros((9,9)); zeros((9,9)) I]
    return eigen(QEVP_LH,QEVP_RH)
end

function getQEPpoly4(H_inv, k_1, k_2, k_x, k_y, V_2, V)
    ζ = (k_1^2-k_2^2) * V_2 / V / k_1^2
    absGs = dropdims(sqrt.(sum(Gs.^2,dims=1)),dims=1)
    Gzs = Gs[3,:,:,:]
    uuu = Gzs*l.R
    IPvec = [sin.(uuu)./uuu,
             1im ./ uuu.^2 .* (sin.(uuu) - uuu.*cos.(uuu)),
             1   ./ uuu.^3 .* (2*uuu.*cos.(uuu) + (uuu.^2 .- 2).*sin.(uuu)),
             1im ./ uuu.^4 .* ((6*uuu .- uuu.^3).*cos.(uuu) + (3*uuu.^2 .- 6).*sin.(uuu)),
             1   ./ uuu.^5 .* (4*(uuu.^3 .-6*uuu).*cos.(uuu) + (uuu.^4 .- 12*uuu.^2 .+ 24).*sin.(uuu))]
    IPvec = [IPvec[n][i,j,k] for n in 1:5, i in 1:2*l.NG+1, j in 1:1, k in 1:1]
    IPvec[:,l.NG+1,:,:] .= 0
    IPvecconj = conj.(IPvec)
    @einsum Pp[i,j,k,n,m] := IPvec[i,k,n,m] * IPvecconj[j,k,n,m]
    Kk = [kron(Pp[:,:,k,n,m],H_inv[:,:,k,n,m]) for k in 1:2*l.NG+1, n in 1:1, m in 1:1]
    Kk = sum(Kk)
    Qq = [1.0+0im 0 1/3 0 1/5;0 1/3 0 1/5 0;1/3 0 1/5 0 1/7;0 1/5 0 1/7 0;1/5 0 1/7 0 1/9]
    Pp0 = [1.0+0im 0 1/3 0 1/5;0 0 0 0 0;1/3 0 1/9 0 1/15;0 0 0 0 0;1/5 0 1/15 0 1/25]
    Kk = kron(Qq,one(ones(3,3))) - (k_1^2-k_2^2)* V_2/V * Kk
    A2 = Kk - ζ * kron(Pp0,[0.0 0 0; 0 0 0; 0 0 1])
    A1 = -ζ * kron(Pp0,(k_x *[0.0 0 1; 0 0 0; 1 0 0] + k_y *[0.0 0 0; 0 0 1; 0 1 0]))
    A0 = ζ * kron(Pp0,(k_1^2 * I - k_x^2 *[1.0 0 0; 0 0 0; 0 0 0] -
        k_y^2 *[0.0 0 0; 0 1 0; 0 0 0] - k_x * k_y *
        [0.0 1 0; 1 0 0; 0 0 0])) - (k_1^2 - k_x^2 - k_y^2) * Kk
    QEVP_LH = [A1 A0; -I zeros((15,15))]
    QEVP_RH = -[A2 zeros((15,15)); zeros((15,15)) I]
    return eigen(QEVP_LH,QEVP_RH)
end

function getQEPpolyx(deg, H_inv, k_1, k_2, k_x, k_y, V_2, V)
    ζ = (k_1^2-k_2^2) * V_2 / V / k_1^2
    absGs = dropdims(sqrt.(sum(Gs.^2,dims=1)),dims=1)
    Gzs = Gs[3,:,:,:]
    uuu = Gzs*l.R
    if deg == 0
        IPvec = sin.(uuu)./uuu
        IPvec[l.NG+1,:,:] .= 0
        Qq = 1
        Pp0 = 1
    elseif deg == 2
        IPvec = [sin.(uuu)./uuu,
                 1im ./ uuu.^2 .* (sin.(uuu) - uuu.*cos.(uuu)),
                 1   ./ uuu.^3 .* (2*uuu.*cos.(uuu) + (uuu.^2 .- 2).*sin.(uuu))]
        IPvec = [IPvec[n][i,j,k] for n in 1:deg+1, i in 1:2*l.NG+1, j in 1:1, k in 1:1]
        IPvec[:,l.NG+1,:,:] .= 0
        Qq = [1.0+0im 0 1/3;0 1/3 0;1/3 0 1/5]
        Pp0 = [1.0+0im 0 1/3;0 0 0;1/3 0 1/9]
    elseif deg == 4
        IPvec = [sin.(uuu)./uuu,
                 1im ./ uuu.^2 .* (sin.(uuu) - uuu.*cos.(uuu)),
                 1   ./ uuu.^3 .* (2*uuu.*cos.(uuu) + (uuu.^2 .- 2).*sin.(uuu)),
                 1im ./ uuu.^4 .* ((6*uuu .- uuu.^3).*cos.(uuu) + (3*uuu.^2 .- 6).*sin.(uuu)),
                 1   ./ uuu.^5 .* (4*(uuu.^3 .-6*uuu).*cos.(uuu) + (uuu.^4 .- 12*uuu.^2 .+ 24).*sin.(uuu))]
        IPvec = [IPvec[n][i,j,k] for n in 1:deg+1, i in 1:2*l.NG+1, j in 1:1, k in 1:1]
        IPvec[:,l.NG+1,:,:] .= 0
        Qq = [1.0+0im 0 1/3 0 1/5;0 1/3 0 1/5 0;1/3 0 1/5 0 1/7;0 1/5 0 1/7 0;1/5 0 1/7 0 1/9]
        Pp0 = [1.0+0im 0 1/3 0 1/5;0 0 0 0 0;1/3 0 1/9 0 1/15;0 0 0 0 0;1/5 0 1/15 0 1/25]
    else
        println("specified polynomial degree not implemented")
    end
    IPvecconj = conj.(IPvec)
    if deg == 0
        Pp = IPvec.*IPvecconj
        Kk = [Pp[k,n,m]*H_inv[:,:,k,n,m] for k in 1:2*l.NG+1, n in 1:1, m in 1:1]
    else
        @einsum Pp[i,j,k,n,m] := IPvec[i,k,n,m] * IPvecconj[j,k,n,m]
        Kk = [kron(Pp[:,:,k,n,m],H_inv[:,:,k,n,m]) for k in 1:2*l.NG+1, n in 1:1, m in 1:1]
    end
    Kk = sum(Kk)
    Kk = kron(Qq,one(ones(3,3))) - (k_1^2-k_2^2)* V_2/V * Kk
    A2 = Kk - ζ * kron(Pp0,[0.0 0 0; 0 0 0; 0 0 1])
    A1 = -ζ * kron(Pp0,(k_x *[0.0 0 1; 0 0 0; 1 0 0] + k_y *[0.0 0 0; 0 0 1; 0 1 0]))
    A0 = ζ * kron(Pp0,(k_1^2 * I - k_x^2 *[1.0 0 0; 0 0 0; 0 0 0] -
        k_y^2 *[0.0 0 0; 0 1 0; 0 0 0] - k_x * k_y *
        [0.0 1 0; 1 0 0; 0 0 0])) - (k_1^2 - k_x^2 - k_y^2) * Kk
    QEVP_LH = [A1 A0; -I zeros(3*(deg+1),3*(deg+1))]
    QEVP_RH = -[A2 zeros(3*(deg+1),3*(deg+1)); zeros(3*(deg+1),3*(deg+1)) I]
    return eigen(QEVP_LH,QEVP_RH)
end

function getCuEP(deg, H_inv, k_1, k_2, k_x, k_y, V_2, V)
    ζ = (k_1^2-k_2^2) * V_2 / V / k_1^2
    Gzs = Gs[3,:,:,:]
    uuu = Gzs*l.R
    if deg == 0
        IPvec = sin.(uuu)./uuu
        Qq = 0
        Pp0 = 1
    elseif deg == 2
        IPvec = [sin.(uuu)./uuu,
                 1im ./ uuu.^2 .* (sin.(uuu) - uuu.*cos.(uuu)),
                 1   ./ uuu.^3 .* (2*uuu.*cos.(uuu) + (uuu.^2 .- 2).*sin.(uuu))]
        Qq = [1.0+0im 0 1/3;0 1/3 0;1/3 0 1/5]
        Pp0 = [1.0+0im 0 1/3;0 0 0;1/3 0 1/9]
    elseif deg == 4
        IPvec = [sin.(uuu)./uuu,
                 1im ./ uuu.^2 .* (sin.(uuu) - uuu.*cos.(uuu)),
                 1   ./ uuu.^3 .* (2*uuu.*cos.(uuu) + (uuu.^2 .- 2).*sin.(uuu)),
                 1im ./ uuu.^4 .* ((6*uuu .- uuu.^3).*cos.(uuu) + (3*uuu.^2 .- 6).*sin.(uuu)),
                 1   ./ uuu.^5 .* (4*(uuu.^3 .-6*uuu).*cos.(uuu) + (uuu.^4 .- 12*uuu.^2 .+ 24).*sin.(uuu))]
        Qq = [1.0+0im 0 1/3 0 1/5;0 1/3 0 1/5 0;1/3 0 1/5 0 1/7;0 1/5 0 1/7 0;1/5 0 1/7 0 1/9]
        Pp0 = [1.0+0im 0 1/3 0 1/5;0 0 0 0 0;1/3 0 1/9 0 1/15;0 0 0 0 0;1/5 0 1/15 0 1/25]
    else
        println("specified polynomial degree not implemented")
    end
    IPvec = [IPvec[n][i,j,k] for n in 1:deg+1, i in 1:2*l.NG+1, j in 1:1, k in 1:1]
    IPvec[:,l.NG+1,:,:] .= 0
    IPvecconj = conj.(IPvec)
    @einsum Pp[i,j,k,n,m] := IPvec[i,k,n,m] * IPvecconj[j,k,n,m]
    kpar = [k_x,k_y,0]
    kpar² = dot(kpar,kpar)
    kparTx = kpar*kpar'
    kparG = kpar .+ Gs
    @einsum kparG²[k,n,m] := kparG[i,k,n,m]*kparG[i,k,n,m]
    @einsum kparGTx[i,j,k,n,m] := kparG[i,k,n,m]*kparG[j,k,n,m]
    Rr0 = [kron(Pp[:,:,k,n,m],(k_1^2*one(ones(3,3)) .- kparGTx)[:,:,k,n,m]) for k in 1:2*l.NG+1, n in 1:1, m in 1:1]
    Ss0 = (k_1^2 - kpar²)./(k_1^2 .- kparG²) .* Rr0
    Ss0 = ζ * sum(Ss0)
    e3 = [0,0,1]
    @einsum kparGxe3[i,j,k,n,m] := kparG[i,k,n,m]*e3[j]
    Rr1 = [kron(Pp[:,:,k,n,m],(k_1^2*one(ones(3,3)) .- kparGTx .- kparGxe3 .- transpose.(conj.(kparGxe3)))[:,:,k,n,m]) for k in 1:2*l.NG+1, n in 1:1, m in 1:1]
    Ss1 = (k_1^2 - kpar²)./(k_1^2 .- kparG²) .* (2*Gzs ./ (k_1^2 .- kparG²)) .* Rr1
    Ss1 = ζ * sum(Ss1)
    Rr2 = [kron(Pp[:,:,k,n,m],(k_1^2*one(ones(3,3)) .- kparGTx)[:,:,k,n,m]) for k in 1:2*l.NG+1, n in 1:1, m in 1:1]
    Ss2 = 1 ./(k_1^2 .- kparG²) .* Rr2
    Ss2 = ζ * sum(Ss2)
    Rr3 = [kron(Pp[:,:,k,n,m],(k_1^2*one(ones(3,3)) .- kparGTx .- kparGxe3 .- transpose.(conj.(kparGxe3)))[:,:,k,n,m]) for k in 1:2*l.NG+1, n in 1:1, m in 1:1]
    Ss3 = 1 ./(k_1^2 .- kparG²) .* (2*Gzs ./ (k_1^2 .- kparG²)) .* Rr3
    Ss3 = ζ * sum(Ss3)
    Mm0 = (k_1^2-kpar²) * kron(Qq,one(ones(3,3))) - ζ* kron(Pp0,k_1^2*one(ones(3,3))-kparTx) - ζ * Ss0
    Mm1 = ζ* kron(Pp0,kpar*[0,0,1]' .- transpose(conj.(kpar*[0,0,1]'))) - ζ * Ss1
    Mm2 = -kron(Qq,one(ones(3,3))) .+ ζ * kron(Pp0,[0 0 0;0 0 0;0 0 1]) .+ Ss2
    Mm3 = Ss3

    nep = PEP([Mm0, Mm1, Mm2, Mm3])
    λ, v = polyeig(nep)

    return λ, v
end

function getM(λ_value, IPs, eps = 1.0e-8)::Array{Complex{Float64},2}

    k_v = [p.k_x ; p.k_y; λ_value]
    H_inv = getHinv(Gs, k_v, p.k_1)
    @einsum GH_sum[i,j,k,n,m] := IPs[k,n,m] * H_inv[i,j,k,n,m]
    GH_sum = dropdims(sum(GH_sum, dims=(3,4,5)),dims=(3,4,5))
    return I - GH_sum
end

function getMder(λ_value, IPs, eps = 1.0e-8)::Complex{Float64}

    return (det(getM(λ_value+eps, IPs)) -
            det(getM(λ_value-eps, IPs)))/(2*eps)
end

function scalarNewton(init, maxiter=1000, tol2=5e-9)

    k = init
    for nn in 1:maxiter
        phik = det(getM(k, IP²_factor))
        knew = k - phik / getMder(k, IP²_factor)
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

function getpoly2M(λ_value, eps = 1.0e-8)::Array{Complex{Float64},2}
    k_v = [p.k_x ; p.k_y; λ_value]
    H_inv = getHinv(Gs, k_v, p.k_1)
    Gzs = Gs[3,:,:,:]
    IPvec = [sinc.(V_2/2/pi * Gzs),
        1im ./(V_2 * Gzs) .* (sinc.(V_2/2/pi * Gzs) - cos.(V_2/2 * Gzs)),
        1/4 * sinc.(V_2/2/pi * Gzs) + 2 ./(V_2^2 * Gzs.^2) .* (cos.(V_2/2 * Gzs) - sinc.(V_2/2/pi * Gzs))]
    IPvec = [IPvec[n][i,j,k] for n in 1:3, i in 1:2*l.NG+1, j in 1:1, k in 1:1]
    IPvec[:,l.NG+1,1,1] = [1.0,0,1/12]
    IPvecconj = conj.(IPvec)
    @einsum Pp[i,j,k,n,m] := IPvec[i,k,n,m] * IPvecconj[j,k,n,m]
    summands = [kron(Pp[:,:,k,n,m],H_inv[:,:,k,n,m]) for k in 1:2*l.NG+1, n in 1:1, m in 1:1]
    latsum = sum(summands)
    Qq = [1.0+0im 0 1/12;0 1/12 0;1/12 0 1/80]
    EVPmatrix = kron(Qq,one(ones(3,3))) - ((p.k_1^2-p.k_2^2) * l.V_2 / l.V) * latsum
    return EVPmatrix
end

function getpoly2Mder(λ_value, eps = 1.0e-8)::Complex{Float64}

    return (det(getpoly2M(λ_value+eps)) -
            det(getpoly2M(λ_value-eps)))/(2*eps)
end

function poly2Newton(init, maxiter=1000, tol2=5e-9)

    k = init
    for nn in 1:maxiter
        phik = det(getpoly2M(k))
        knew = k - phik / getpoly2Mder(k)
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

function getpoly4M(λ_value, eps = 1.0e-8)::Array{Complex{Float64},2}
    k_v = [p.k_x ; p.k_y; λ_value]
    H_inv = getHinv(Gs, k_v, p.k_1)
    Gzs = Gs[3,:,:,:]
    uuu = Gzs*l.R
    IPvec = [sin.(uuu)./uuu,
             1im ./ uuu.^2 .* (sin.(uuu) - uuu.*cos.(uuu)),
             1   ./ uuu.^3 .* (2*uuu.*cos.(uuu) + (uuu.^2 .- 2).*sin.(uuu)),
             1im ./ uuu.^4 .* ((6*uuu .- uuu.^3).*cos.(uuu) + (3*uuu.^2 .- 6).*sin.(uuu)),
             1   ./ uuu.^5 .* (4*(uuu.^3 .-6*uuu).*cos.(uuu) + (uuu.^4 .- 12*uuu.^2 .+ 24).*sin.(uuu))]
    # IPvec = [sinc.(V_2/2/pi * Gzs),
    #     1im ./(V_2 * Gzs).^2 .* (-2*(V_2 * Gzs)    .*cos.(V_2/2*Gzs) + 4*                  sin.(V_2/2*Gzs)),
    #     1 ./(V_2 * Gzs).^3   .* (2*(V_2 * Gzs).^2  .*sin.(V_2/2*Gzs) + 8*(V_2 * Gzs)     .*cos.(V_2/2*Gzs) - 16*                 sin.(V_2/2*Gzs)),
    #     1im ./(V_2 * Gzs).^4 .* (-2*(V_2 * Gzs).^3 .*cos.(V_2/2*Gzs) + 12*(V_2 * Gzs).^2 .*sin.(V_2/2*Gzs) + 48*(V_2 * Gzs)    .*cos.(V_2/2*Gzs) - 96*               sin.(V_2/2*Gzs)),
    #     1 ./(V_2 * Gzs).^5   .* (2*(V_2 * Gzs).^4  .*sin.(V_2/2*Gzs) + 16*(V_2 * Gzs).^3 .*cos.(V_2/2*Gzs) - 96*(V_2 * Gzs).^2 .*sin.(V_2/2*Gzs) - 384*(V_2 * Gzs) .*cos.(V_2/2*Gzs) + 768*sin.(V_2/2*Gzs))]
    IPvec = [IPvec[n][i,j,k] for n in 1:5, i in 1:2*l.NG+1, j in 1:1, k in 1:1]
    IPvec[:,l.NG+1,1,1] = [1.0,0,1/3,0,1/5]
    IPvecconj = conj.(IPvec)
    @einsum Pp[i,j,k,n,m] := IPvec[i,k,n,m] * IPvecconj[j,k,n,m]
    summands = [kron(Pp[:,:,k,n,m],H_inv[:,:,k,n,m]) for k in 1:2*l.NG+1, n in 1:1, m in 1:1]
    latsum = sum(summands)
    Qq = [1.0+0im 0 1/3 0 1/5;0 1/3 0 1/5 0;1/3 0 1/5 0 1/7;0 1/5 0 1/7 0;1/5 0 1/7 0 1/9]
    EVPmatrix = kron(Qq,one(ones(3,3))) - ((p.k_1^2-p.k_2^2) * l.V_2 / l.V) * latsum
    return EVPmatrix
end

function getpoly4Mder(λ_value, eps = 1.0e-8)::Complex{Float64}

    return (det(getpoly4M(λ_value+eps)) -
            det(getpoly4M(λ_value-eps)))/(2*eps)
end

function poly4Newton(init, maxiter=1000, tol2=5e-9)

    k = init
    for nn in 1:maxiter
        phik = det(getpoly4M(k))
        knew = k - phik / getpoly4Mder(k)
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

function getpolyxM(deg, λ_value, eps = 1.0e-8)::Array{Complex{Float64},2}
    k_v = [p.k_x ; p.k_y; λ_value]
    H_inv = getHinv(Gs, k_v, p.k_1)
    Gzs = Gs[3,:,:,:]
    uuu = Gzs*l.R
    if deg == 0
        IPvec = sin.(uuu)./uuu
        Qq = 1
        IPvec[l.NG+1,:,:] .= 1
    elseif deg == 2
        IPvec = [sin.(uuu)./uuu,
                 1im ./ uuu.^2 .* (sin.(uuu) - uuu.*cos.(uuu)),
                 1   ./ uuu.^3 .* (2*uuu.*cos.(uuu) + (uuu.^2 .- 2).*sin.(uuu))]
        Qq = [1.0+0im 0 1/3;0 1/3 0;1/3 0 1/5]
        IPvec = [IPvec[n][i,j,k] for n in 1:(deg+1), i in 1:2*l.NG+1, j in 1:1, k in 1:1]
        IPvec[:,l.NG+1,:,:] .= [1.0,0,1/3]
    elseif deg == 4
        IPvec = [sin.(uuu)./uuu,
                 1im ./ uuu.^2 .* (sin.(uuu) - uuu.*cos.(uuu)),
                 1   ./ uuu.^3 .* (2*uuu.*cos.(uuu) + (uuu.^2 .- 2).*sin.(uuu)),
                 1im ./ uuu.^4 .* ((6*uuu .- uuu.^3).*cos.(uuu) + (3*uuu.^2 .- 6).*sin.(uuu)),
                 1   ./ uuu.^5 .* (4*(uuu.^3 .-6*uuu).*cos.(uuu) + (uuu.^4 .- 12*uuu.^2 .+ 24).*sin.(uuu))]
        Qq = [1.0+0im 0 1/3 0 1/5;0 1/3 0 1/5 0;1/3 0 1/5 0 1/7;0 1/5 0 1/7 0;1/5 0 1/7 0 1/9]
        IPvec = [IPvec[n][i,j,k] for n in 1:(deg+1), i in 1:2*l.NG+1, j in 1:1, k in 1:1]
        IPvec[:,l.NG+1,:,:] .= [1.0,0,1/3,0,1/5]
    else
        println("specified polynomial degree not implemented")
    end
    IPvecconj = conj.(IPvec)
    if deg == 0
        Pp = IPvec.*IPvecconj
        summands = [Pp[k,n,m]*H_inv[:,:,k,n,m] for k in 1:2*l.NG+1, n in 1:1, m in 1:1]
    else
        @einsum Pp[i,j,k,n,m] := IPvec[i,k,n,m] * IPvecconj[j,k,n,m]
        summands = [kron(Pp[:,:,k,n,m],H_inv[:,:,k,n,m]) for k in 1:2*l.NG+1, n in 1:1, m in 1:1]
    end
    latsum = sum(summands)
    EVPmatrix = kron(Qq,one(ones(3,3))) - ((p.k_1^2-p.k_2^2) * l.V_2 / l.V) * latsum
    return EVPmatrix
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

function getFieldValue(H, KG_slice_y, KG_slice_z, y, z)::ComplexF64

    return sum(H .* exp.(1im*KG_slice_y*y) .* exp.(1im*KG_slice_z*-z ))
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
