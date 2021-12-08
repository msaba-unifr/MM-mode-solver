include("functions.jl")

function Init_Workspace(; λ = 600, φ = 45, θ = 45, NG = 10, ϵ_1 = 1 + 0im,
        ϵ_2 = "Ag_JC_nk.txt", A = [30/2 30; sqrt(3)*30/2 0],
        Rad = 10.0, mmdim = 2)

    #Interpolating (n,k) data from txt file
    if typeof(ϵ_2) == String
        file_loc = string(pwd(), "\\MaterialModels\\", ϵ_2)
        global mat_file = ϵ_2
        eps_data = readdlm(file_loc, '\t', Float64, '\n')
        itp1 = LinearInterpolation(eps_data[:,1], eps_data[:,2])
        itp2 = LinearInterpolation(eps_data[:,1], eps_data[:,3])
        ϵ_2 = (itp1.(λ ./ 1000) + itp2.(λ ./ 1000) * 1im).^2
    end

    return Lattice(NG,A,Rad), Parameters(λ, φ, θ, ϵ_1, ϵ_2)
end

function getpolyxMode(deg, l::Lattice, p::Parameters; manual_ks=[0im,0im])

    ks = zeros(ComplexF64,(2))
    dim = 3*((deg[1]+1)*(deg[2]+1))
    if norm(manual_ks) != 0
        ks = manual_ks
    end

    #Solve NLEVP for each non filtered mode
    ksols = zeros(ComplexF64,(2))
    csols = zeros(ComplexF64,(dim,2))
    for mode = 1:1 #CHANGED to 1:1 from 1:2 for BM
        k_sol = polyxNewton(deg,ks[mode], l, p)
        c_sol = qr(transpose(conj(getpolyxM(deg,k_sol, l, p))), Val(true)).Q[:,end]
        ksols[mode] = k_sol
        csols[:, mode] = c_sol
    end
    return ksols, csols
end
