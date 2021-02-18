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
        file_loc = string(pwd(), "\\MaterialModels\\", mat_file)
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

function update_dependencies!(; kwargs...)

    vars = keys(kwargs)
    for var in vars
        if var == :wl
            global p = Parameters(kwargs[var], p.azim, p.incl, p.e_m, p.e_bg)
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
                file_loc = string(pwd(), "\\MaterialModels\\", mat_file)
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

function test(; kwargs...)

    return kwargs
end
