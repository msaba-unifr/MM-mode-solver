include("functions.jl")

function Init_Workspace(; wl = 600, φ = 45, θ = 45, NG = 10, ϵ_bg = 1 + 0im,
        ϵ_m = "Ag_JC_nk.txt", A = [30/2 30; sqrt(3)*30/2 0],
        Rad = 10.0, V_2 = pi*10.0^2)
    global l = Lattice2D(NG,A,V_2,Rad)
    #Creating G_space
    global Gs = getGspace()
    global IP²_noDC = dropdims(InnerProd.(sqrt.(sum(Gs.*Gs,dims=1))*Rad,
        exclude_DC=true),dims=1).^2
    global IP = dropdims(InnerProd.(sqrt.(sum(Gs.*Gs,dims=1))*Rad),dims=1)
    global IP² = IP.^2

    #Interpolating (n,k) data from txt file
    if typeof(ϵ_m) == String
        file_loc = string(pwd(), "\\MaterialModels\\", ϵ_m)
        global mat_file = ϵ_m
        eps_data = readdlm(file_loc, '\t', Float64, '\n')
        itp1 = LinearInterpolation(eps_data[:,1], eps_data[:,2])
        itp2 = LinearInterpolation(eps_data[:,1], eps_data[:,3])
        ϵ_m = (itp1.(wl ./ 1000) + itp2.(wl ./ 1000) * 1im).^2
    end

    global p = Parameters(wl, φ, θ, ϵ_m, ϵ_bg)
    return
end

function getMode()

    o_vec = zeros(ComplexF64, (3,1))
    𝓗invs = getHinv(Gs,o_vec)

    #InitialGuess
    eigs_init = getInitGuess(IP²_noDC,𝓗invs)
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
        c_sol = qr(conj(getMder(k_sol,0)), Val(true)).Q[:,end]
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
    HikG = getHinv(Gs, [p.k_x, p.k_y, k_sol])
    absGs = dropdims(sqrt.(sum(Gs.^2,dims=1)),dims=1)
    #Calculation according to manuscript
    @tensor H_c[i,k,n,m] := IP[k,n,m] * HikG[i,j,k,n,m] * c_sol[j]
    H_c = H_c ./ l.V
    #Field components for every z-y position in image range
    E = [sum(H_c[idx,:,:,:] .* exp.(1im*kpGs[2,:,:,:]*y) .*
        exp.(1im*kpGs[3,:,:,:]*-z )) for idx in 1:3, z in zs, y in ys]
    return E
end

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
            global l = Lattice2D(l.NG, l.A, l.V_2, kwargs[var])
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