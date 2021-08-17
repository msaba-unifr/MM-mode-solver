
function update_dependencies!(; kwargs...)
    vars = keys(kwargs)
    for var in vars
        if var == :wl
            global p = Parameters(kwargs[var], p.azim, p.incl, p.e_m, p.e_bg)
            update_dependencies!(ϵ_m = mat_file)
        end
        if var == :φ
            global p = Parameters(p.lambda, kwargs[var], p.incl, p.e_m, p.e_bg)
        end
        if var == :θ
            global p = Parameters(p.lambda, p.azim, kwargs[var], p.e_m, p.e_bg)
        end
        if var == :NG
            global l = Lattice2D(kwargs[var],l.A,l.V_2, l.R)
            #Creating G_space
            global Gs = getGspace()
            global IP²_noDC = dropdims(InnerProd.(sqrt.(sum(Gs.*Gs,dims=1))*l.R,
                exclude_DC=true),dims=1).^2
            global IP = dropdims(InnerProd.(sqrt.(sum(Gs.*Gs,dims=1))*l.R),dims=1)
            global IP² = IP.^2
        end
        if var == :ϵ_bg
            global p = Parameters(p.lambda, p.azim, p.incl,  p.e_m, kwargs[var])
        end
        if var == :ϵ_m
            if typeof(kwargs[var]) == String
                file_loc = string(pwd(), "\\MaterialModels\\", kwargs[var])
                eps_data = readdlm(file_loc, '\t', Float64, '\n')
                itp1 = LinearInterpolation(eps_data[:,1], eps_data[:,2])
                itp2 = LinearInterpolation(eps_data[:,1], eps_data[:,3])
                ϵ_m = (itp1.(p.lambda ./ 1000) + itp2.(p.lambda ./ 1000) * 1im).^2
                global p = Parameters(p.lambda, p.azim, p.incl,  ϵ_m, p.e_bg)
            else
                global p = Parameters(p.lambda, p.azim, p.incl,  kwargs[var],
                    p.e_bg)
            end
        end
        if var == :A
            global l = Lattice2D(l.NG, kwargs[var], l.V_2, l.R)
            #Creating G_space
            global Gs = getGspace()
            global IP²_noDC = dropdims(InnerProd.(sqrt.(sum(Gs.*Gs,dims=1))*l.R,
                exclude_DC=true),dims=1).^2
            global IP = dropdims(InnerProd.(sqrt.(sum(Gs.*Gs,dims=1))*l.R),dims=1)
            global IP² = IP.^2
        end
        if var == :Rad
            global l = Lattice2D(l.NG, l.A, pi*kwargs[var]^2, kwargs[var])
            global IP²_noDC = dropdims(InnerProd.(sqrt.(sum(Gs.*Gs,dims=1))*
                kwargs[var], exclude_DC=true),dims=1).^2
            global IP = dropdims(InnerProd.(sqrt.(sum(Gs.*Gs,dims=1))*
                kwargs[var]),dims=1)
            global IP² = IP.^2
        end
        if var == :V_2
            global l = Lattice2D(l.NG, l.A, kwargs[var], l.R)
        end
    end
end

function getMode()

    o_vec = zeros(ComplexF64, (3,1))
    𝓗invs = getHinv(Gs,o_vec, p.k_1)
    #InitialGuess
    eigs_init = getInitGuess(IP²_noDC,𝓗invs, p.k_1, p.k_2, p.k_x, p.k_y,
        l.V_2, l.V)
    λs,vs = eigs_init.values, eigs_init.vectors

    ks = zeros(ComplexF64,(2))
    cs = zeros(ComplexF64,(3,2))
    for (i, λ_val) in enumerate(λs)

        if real(λ_val) <= 0
            continue
        elseif abs(1- (λ_val^2+p.k_x^2+p.k_y^2)/(p.e_bg*p.k_0^2)) < 1e-8
            continue
        end
        if ks[1] == 0
            ks[1] = λs[i]
            cs[:, 1] = vs[4:6, i]
        else
            ks[2] = λs[i]
            cs[:, 2] = vs[4:6, i]
        end
    end

    #Solve NLEVP for each non filtered mode
    ksols = zeros(ComplexF64,(2))
    csols = zeros(ComplexF64,(3,2))
    for mode = 1:2

        global IP²_factor = (p.k_1^2 - p.k_2^2) / l.V_2 / l.V .* IP²
        k_sol = scalarNewton(ks[mode])
        c_sol = qr(transpose(conj(getM(k_sol, IP²_factor))), Val(true)).Q[:,end]
        ksols[mode] = k_sol
        csols[:, mode] = c_sol
    end
    return ksols, csols
end

function getpoly2Mode()

    o_vec = zeros(ComplexF64, (3,1))
    𝓗invs = getHinv(Gs,o_vec, p.k_1)
    #InitialGuess
    eigs_init = getQEPpoly2(𝓗invs, p.k_1, p.k_2, p.k_x, p.k_y,
        l.V_2, l.V)
    λs,vs = eigs_init.values, eigs_init.vectors

    ks = zeros(ComplexF64,(2))
    cs = zeros(ComplexF64,(9,2))
    for (i, λ_val) in enumerate(λs)

        if real(λ_val) <= 0
            continue
        elseif abs(1- (λ_val^2+p.k_x^2+p.k_y^2)/(p.e_bg*p.k_0^2)) < 1e-8
            continue
        end
        if ks[1] == 0
            ks[1] = λs[i]
            cs[:, 1] = vs[10:18, i]
        else
            ks[2] = λs[i]
            cs[:, 2] = vs[10:18, i]
        end
    end

    #Solve NLEVP for each non filtered mode
    ksols = zeros(ComplexF64,(2))
    csols = zeros(ComplexF64,(9,2))
    for mode = 1:2

        # global IP²_factor = (p.k_1^2 - p.k_2^2) / l.V_2 / l.V .* IP²
        k_sol = poly2Newton(ks[mode])
        c_sol = qr(transpose(conj(getpoly2M(k_sol))), Val(true)).Q[:,end]
        ksols[mode] = k_sol
        csols[:, mode] = c_sol
    end
    return ksols, csols
end

function getpoly4Mode()

    o_vec = zeros(ComplexF64, (3,1))
    𝓗invs = getHinv(Gs,o_vec, p.k_1)
    #InitialGuess
    eigs_init = getQEPpoly4(𝓗invs, p.k_1, p.k_2, p.k_x, p.k_y,
        l.V_2, l.V)
    λs,vs = eigs_init.values, eigs_init.vectors

    ks = zeros(ComplexF64,(2))
    cs = zeros(ComplexF64,(15,2))
    for (i, λ_val) in enumerate(λs)

        if real(λ_val) <= 0
            continue
        elseif abs(1- (λ_val^2+p.k_x^2+p.k_y^2)/(p.e_bg*p.k_0^2)) < 1e-8
            continue
        end
        if ks[1] == 0
            ks[1] = λs[i]
            cs[:, 1] = vs[16:30, i]
        else
            ks[2] = λs[i]
            cs[:, 2] = vs[16:30, i]
        end
    end

    #Solve NLEVP for each non filtered mode
    ksols = zeros(ComplexF64,(2))
    csols = zeros(ComplexF64,(15,2))
    for mode = 1:2

        # global IP²_factor = (p.k_1^2 - p.k_2^2) / l.V_2 / l.V .* IP²
        k_sol = poly4Newton(ks[mode])
        c_sol = qr(transpose(conj(getpoly4M(k_sol))), Val(true)).Q[:,end]
        ksols[mode] = k_sol
        csols[:, mode] = c_sol
    end
    return ksols, csols
end

function getE_Field(k_sol, c_sol, img_yrange, img_zrange, res)

    ys = -img_yrange/2 : res : img_yrange/2
    zs = -img_zrange/2 : res : img_zrange/2
    #Precomputing variables
    kpGs = [p.k_x, p.k_y, k_sol] .+ Gs
    HikG = getHinv(Gs, [p.k_x, p.k_y, k_sol], p.k_1)
    absGs = dropdims(sqrt.(sum(Gs.^2,dims=1)),dims=1)
    #Calculation according to manuscript
    IPs = IP
    @einsum H_c[i,k,n,m] := IPs[k,n,m] * HikG[i,j,k,n,m] * c_sol[j]
    H_c = H_c ./ l.V
    #Field components for every z-y position in image range
    kpGs_y = kpGs[2,:,:,:]
    kpGs_z = kpGs[3,:,:,:]
    H_c = (H_c[1,:,:,:], H_c[2,:,:,:], H_c[3,:,:,:])
    E = [getFieldValue(H_c[idx], kpGs_y, kpGs_z, y, z)
        for idx in 1:3, z in zs, y in ys]
    return E
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
