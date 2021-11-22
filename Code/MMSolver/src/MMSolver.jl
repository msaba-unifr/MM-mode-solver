module MMSolver

export init_workspace, get_polyx_mode, getE_Field, get_single_mode

using Distributed
using LinearAlgebra
using Einsum
using SpecialFunctions
using DelimitedFiles
using Interpolations

include("parameters.jl")
include("functions.jl")

function init_workspace(; λ = 600, φ = 45, θ = 45, NG = 10, ϵ_1 = 1 + 0im,
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

function get_polyx_mode(deg,l::Lattice,p::Parameters;manual_ks=[0im,0im])
    ks = zeros(ComplexF64,(2))
    dim = 3*((deg[1]+1)*(deg[2]+1))
    if norm(manual_ks) != 0
        ks = manual_ks
    end
    #Solve NLEVP for each non filtered mode
    ksols = zeros(ComplexF64,(2))
    csols = zeros(ComplexF64,(dim,2))
    Niters = zeros(Int,(2))
    for mode = 1:2
        k_sol,Niters[mode] = polyxNewton(deg,ks[mode],l,p)
        c_sol = qr(transpose(conj(getpolyxM(deg,k_sol,l,p))), Val(true)).Q[:,end]
        ksols[mode] = k_sol
        csols[:, mode] = c_sol
    end
    return ksols, csols, Niters
end

function get_single_mode(deg,l::Lattice,p::Parameters;manual_ks)
    ks = 0im
    dim = 3*((deg[1]+1)*(deg[2]+1))
    if norm(manual_ks) != 0
        ks = manual_ks
    end
    println(ks)
    #Solve NLEVP for each non filtered mode
    ksols = 0im
    csols = zeros(ComplexF64,(dim,1))
    Niters = 0
    ksols,Niters = polyxNewton(deg,ks,l,p)
    csols = qr(transpose(conj(getpolyxM(deg,ksols,l,p))), Val(true)).Q[:,end]
    return ksols, csols, Niters
end

function getE_Field(polydegs, l, p, k_sol, c_sol, img_yrange, img_zrange, res)
    Qq, Pp0, deg_list = MMSolver.getQq(polydegs)
    ys = -img_yrange/2 : res : img_yrange/2
    zs = -2*img_zrange/4 : res : 2*img_zrange/4
    len_ys = length(collect(ys))
    k_v = [p.k_x, p.k_y, k_sol]
    E = zeros(ComplexF64,(3,length(ys),length(zs)))
    for (ny,y) in enumerate(collect(ys))
        println(ny,"/",len_ys)
        for (nz,z) in enumerate(collect(zs))
            latsum::Array{ComplexF64,2} = @distributed (+) for G in l.Gs
                Efield_IP_summand(k_v,c_sol,G,polydegs,deg_list,l,p) * exp(1im*(k_v+G)[2]*y) * exp(1im*(k_v+G)[3]*z)
            end
            E[:,ny,nz] = latsum
        end
    end
    return E
end

end ## module
