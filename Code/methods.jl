include("functions.jl")

function Init_Workspace(; λ = 600, φ = 45, θ = 45, NG = 10, ϵ_1 = 1 + 0im,
        ϵ_2 = "Ag_JC_nk.txt", A = [30/2 30; sqrt(3)*30/2 0],
        Rad = 10.0, mmdim = 2)
    if mmdim == 1
        global l = Lattice1D(NG,a,Rad)
    elseif mmdim == 2
        global l = Lattice2D(NG,A,Rad)
    end
    #Creating G_space
    global Gs = getGspace(mmdim)

    #Interpolating (n,k) data from txt file
    if typeof(ϵ_2) == String
        file_loc = string(pwd(), "\\MaterialModels\\", ϵ_2)
        global mat_file = ϵ_2
        eps_data = readdlm(file_loc, '\t', Float64, '\n')
        itp1 = LinearInterpolation(eps_data[:,1], eps_data[:,2])
        itp2 = LinearInterpolation(eps_data[:,1], eps_data[:,3])
        ϵ_2 = (itp1.(λ ./ 1000) + itp2.(λ ./ 1000) * 1im).^2
    end

    global p = Parameters(wl, φ, θ, ϵ_1, ϵ_2)
    return
end

function getpolyxMode(deg;oldQEP=false)

    o_vec = zeros(ComplexF64, (3,1))
    𝓗invs = getHinv(Gs,o_vec, p.k_1)
    #InitialGuess
    if oldQEP == true
        eigs_init = getInitGuess(IP²_noDC,𝓗invs, p.k_1, p.k_2, p.k_x, p.k_y,
        l.V_2, l.V)
        QEPdim = 3
    else
        eigs_init = getQEPpolyx(deg,𝓗invs, p.k_1, p.k_2, p.k_x, p.k_y,
        l.V_2, l.V)
        QEPdim = 3*((deg[1]+1)*(deg[2]+1))
    end
    λs,vs = eigs_init.values, eigs_init.vectors

    ks = zeros(ComplexF64,(2))
    cs = zeros(ComplexF64,(QEPdim,2))

    for (i, λ_val) in enumerate(λs)

        if real(λ_val) <= 0
            continue
        elseif abs(1- (λ_val^2+p.k_x^2+p.k_y^2)/(p.eps_1*p.k_0^2)) < 1e-8
            continue
        end
        if ks[1] == 0
            ks[1] = λs[i]
            cs[:, 1] = vs[QEPdim+1:2*QEPdim, i]
        else
            ks[2] = λs[i]
            cs[:, 2] = vs[QEPdim+1:2*QEPdim, i]
        end
    end

    #Solve NLEVP for each non filtered mode
    ksols = zeros(ComplexF64,(2))
    csols = zeros(ComplexF64,(3*((deg[1]+1)*(deg[2]+1)),2))
    for mode = 1:2

        # global IP²_factor = (p.k_1^2 - p.k_2^2) / l.V_2 / l.V .* IP²
        k_sol = polyxNewton(deg,ks[mode])
        c_sol = qr(transpose(conj(getpolyxM(deg,k_sol))), Val(true)).Q[:,end]
        ksols[mode] = k_sol
        csols[:, mode] = c_sol
    end
    return ksols, csols
end
