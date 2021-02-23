using LinearAlgebra
using Einsum
using SpecialFunctions
using NonlinearEigenproblems
using DelimitedFiles
using Interpolations
using Plots
include("methods.jl")

#Parameters set by the user
wl = 700
φ = 90
θ = 0
NG = 30
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

update_dependencies!(NG=100)

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

##########################################
# update_dependencies!(θ=45,φ=0)
# ksols,csols = getMode()
#
# inclsweep = 0 : 0.5 : 90
# kmodes = zeros(ComplexF64,(size(inclsweep,1),2))
# curvecs = zeros(ComplexF64,(3,size(inclsweep,1),2))
# sorttol = 1e-2
# for (nt,th) in enumerate(inclsweep)
#     update_dependencies!(φ=th)
#     kmodes[nt,:], curvecs[:,nt,:] = getMode()
#     if nt != 1 && abs(kmodes[nt, 1] - kmodes[nt-1, 1]) > abs(kmodes[nt, 1] - kmodes[nt-1, end])
#         kmodes[nt, :] = kmodes[nt, end:-1:1]
#     end
# end
#
# incls = collect(inclsweep)
# angles_path = string(pwd(), "\\Data\\anglesweep_const_azim.dat")
# open(angles_path, "w") do io
#            writedlm(io, hcat(real.(incls), real.(kmodes[:,1]), imag.(kmodes[:,1]), real.(kmodes[:,2]), imag(kmodes[:,2])))
# end
###########################################


update_dependencies!(wl=wl)

ksols,csols = getMode()

res = 0.25
yrange = 3*a
zrange = 1.5*sqrt(3)*a
mode = 2
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
heatmap(E_plot)

heatmp = [ [(y-1)*res-yrange/2,(z-1)*res,E_plot[z,y]] for
y in 1:size(E_plot)[2] for
z in 1:size(E_plot)[1] ]

file_path = string(pwd(), "\\Data\\HM_700nm_TE_EI_NI.dat")

open(file_path, "w") do io
    writedlm(io, heatmp)
end
#
# #####################################
# #One Wigner-Seitz
# res = 0.25
# yrange = 3*a/2
# zrange = sqrt(3)*a/2
# mode = 2
# E = getE_Field(ksols[mode], csols[:,mode], yrange, zrange, res)
# E_x = E[1,:,:]
# E_x = abs.(E_x).^2
# E_y = E[2,:,:]
# E_y = abs.(E_y).^2
# E_z = E[3,:,:]
# E_z = abs.(E_z).^2
#
# E_I =  dropdims(sum(abs.(E).^2,dims=1),dims=1)
# pltnrm = maximum(E_I)
#
# E_plot = E_I/pltnrm
# heatmap(E_plot)
#
# heatmpWS = [ [(y-1)*res-yrange/2,(z-1)*res,E_plot[z,y]] for
# y in 1:size(E_plot)[2] for
# z in 1:size(E_plot)[1] ]
#
# for dtpt in heatmpWS
#     if dtpt[2] >= (sqrt(3)*dtpt[1] + 3*sqrt(3)/4*a)
#         dtpt[3]=-1
#     end
# end
#
# for dtpt in heatmpWS
#     if dtpt[2] <= (sqrt(3)*dtpt[1] - sqrt(3)/4*a)
#         dtpt[3]=-1
#     end
# end
#
# file_path = string(pwd(), "\\Data\\WSCell_700nm_TE_EI_NI.dat")
#
# open(file_path, "w") do io
#     writedlm(io, heatmpWS)
# end
