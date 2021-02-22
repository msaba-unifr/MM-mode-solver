using LinearAlgebra
using Einsum
using SpecialFunctions
using NonlinearEigenproblems
using DelimitedFiles
using Interpolations
using Plots
include("methods.jl")

#Parameters set by the user
wl = 361
φ = 90
θ = 0
NG = 20
ϵ_bg = 1 + 0im
mat_file = "Ag_JC_nk.txt"
a = 30.0                            #lattice constant
A = [a/2 a; sqrt(3)*a/2 0]
# IF Ω₂ = {r : |r| < R, r ∈ ℝ², R ∈ ℝ} (i.e. disks/cylinders)
Rad = 10.0
V_2 = pi*Rad^2                              #Volume definition required


#Code starts here
Init_Workspace(wl = wl, φ = φ, θ = θ, NG = NG, ϵ_bg = ϵ_bg,
    ϵ_m = mat_file, A = A, Rad = Rad, V_2 = V_2)



###########################################
# wlsweep = 800 : -0.5 : 300
# kmodes = zeros(ComplexF64,(size(wlsweep,1),2))
# curvecs = zeros(ComplexF64,(3,size(wlsweep,1),2))
# sorttol = 1e-2
# for (nl,λ) in enumerate(wlsweep)
#     update_dependencies!(wl=λ)
#     kmodes[nl,:], curvecs[:,nl,:] = getMode()
#     if nl != 1 && abs(kmodes[nl, 1] - kmodes[nl-1, 1]) > abs(kmodes[nl, 1] - kmodes[nl-1, end])
#         kmodes[nl, :] = kmodes[nl, end:-1:1]
#     end
# end
#
# f_v = 3e5 ./ collect(wlsweep)
# bands_path = string(pwd(), "\\Data\\bandstrct.dat")
# open(bands_path, "w") do io
#            writedlm(io, hcat(real.(f_v), real.(kmodes[:,1]), imag.(kmodes[:,1]), real.(kmodes[:,2]), imag(kmodes[:,2])))
# end
############################################


update_dependencies!(wl=400)

ksols,csols = getMode()

res = 0.5
yrange = 3*a
zrange = 1.5*sqrt(3)*a
mode = 1
E = getE_Field(ksols[mode], csols[:,mode], yrange, zrange, res)
E_x = E[1,:,:]
E_x = abs.(E_x).^2
E_y = E[2,:,:]
E_y = abs.(E_y).^2
E_z = E[3,:,:]
E_z = abs.(E_z).^2

E_I =  dropdims(sum(abs.(E).^2,dims=1),dims=1)
pltnrm = maximum(E_I)

E_plot = E_I/pltnrm
heatmap(E_plot,clims=(0,1))

# heatmp = [ [(y-1)*res-yrange/2,(z-1)*res-zrange/2,E_I[z,y]] for y in 1:size(E_I)[2] for z in 1:size(E_I)[1] ]
#
# file_path = string(pwd(), "\\Data\\test.dat")
#
# open(file_path, "w") do io
#     writedlm(io, heatmp)
# end
