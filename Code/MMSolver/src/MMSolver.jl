module MMSolver

export init_workspace, get_polyx_mode, getE_Field, get_single_mode,
    getC_current_grid

using Distributed
using LinearAlgebra
using Einsum
using SpecialFunctions
using DelimitedFiles
using Interpolations
using ProgressBars

include("parameters.jl")
include("functions.jl")

function init_workspace(; λ = 600, φ = 45, θ = 45, NG = 10, ϵ_1 = 1 + 0im,
        ϵ_2 = "Ag_JC_nk.txt", A = [30/2 30; sqrt(3)*30/2 0],
        Rad = 10.0, mmdim = 2, polydegs = (2,2))

    #Interpolating (n,k) data from txt file
    if typeof(ϵ_2) == String
        file_loc = string(pwd(), "\\MaterialModels\\", ϵ_2)
        global mat_file = ϵ_2
        eps_data = readdlm(file_loc, '\t', Float64, '\n')
        itp1 = LinearInterpolation(eps_data[:,1], eps_data[:,2])
        itp2 = LinearInterpolation(eps_data[:,1], eps_data[:,3])
        ϵ_2 = (itp1.(λ ./ 1000) + itp2.(λ ./ 1000) * 1im).^2
    end

    return Lattice(NG,A,Rad), Parameters(λ, φ, θ, ϵ_1, ϵ_2, polydegs,mat_file)
end

function get_polyx_mode(l::Lattice,p::Parameters;manual_ks=[0im,0im])
    ks = zeros(ComplexF64,(2))
    dim = 3*((p.polydegs[1]+1)*(p.polydegs[2]+1))
    if norm(manual_ks) != 0
        ks = manual_ks
    end
    #Solve NLEVP for each non filtered mode
    ksols = zeros(ComplexF64,(2))
    csols = zeros(ComplexF64,(dim,2))
    Niters = zeros(Int,(2))
    for mode = 1:2
        k_sol,Niters[mode] = polyxNewton(p.polydegs,ks[mode],l,p)
        c_sol = qr(transpose(conj(getpolyxM(p.polydegs,k_sol,l,p))), Val(true)).Q[:,end]
        ksols[mode] = k_sol
        csols[:, mode] = c_sol
    end
    return ksols, csols, Niters
end

function get_single_mode(l::Lattice,p::Parameters;kinit)
    dim = 3*((p.polydegs[1]+1)*(p.polydegs[2]+1))
    #Solve NLEVP for each non filtered mode
    csols = zeros(ComplexF64,(dim,1))
    ksols,Niters = polyxNewton(p.polydegs,kinit,l,p)
    csols = qr(transpose(conj(getpolyxM(p.polydegs,ksols,l,p))), Val(true)).Q[:,end]
    return ksols, csols, Niters
end

function getE_Field(l, p, k_sol, c_sol; img_yrange, img_zrange, res)
    Qq, Pp0, deg_list = MMSolver.getQq(p.polydegs)
    ys = -img_yrange/2 : res : img_yrange/2
    zs = -img_zrange/2 : res : img_zrange/2
    k_v = [p.k_x, p.k_y, k_sol]
    E = zeros(ComplexF64,(3,length(ys),length(zs)))
    for (ny,y) in ProgressBar(enumerate(collect(ys)))
        for (nz,z) in enumerate(collect(zs))
            latsum::Array{ComplexF64,2} = @distributed (+) for G in l.Gs
                Efield_IP_summand(k_v,c_sol,G,p.polydegs,deg_list,l,p) * exp(1im*(k_v+G)[2]*y) * exp(1im*(k_v+G)[3]*z)
            end
            E[:,ny,nz] = latsum
        end
    end
    return E
end

function getC_current_grid(l, p, k_sol, c_sol; Nth, Nr)
    Qq, Pp0, deg_list = getQq(p.polydegs)
    ts = range(0,stop=2*pi,length=Nth)
    rs = range(0,stop=l.R,length=Nr)
    k_v = [p.k_x, p.k_y, k_sol]
    C = zeros(ComplexF64,(3,Nr,Nth))
    for (nr,r) in ProgressBar(enumerate(collect(rs)))
        for (nt,t) in enumerate(collect(ts))
            y = r*cos(t)
            z = r*sin(t)
            C[:,nr,nt] = getC_current_pos(l,p,c_sol,deg_list,k_v,y,z)
        end
    end
    return C
end


end ## module
