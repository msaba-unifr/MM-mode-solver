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
φ = 0
θ = 60
NG = 10
ϵ_bg = 1 + 0im
mat_file = "Ag_JC_nk.txt"
a = 30.0                            #lattice constant
A = [sqrt(2)*a/2 -sqrt(2)*a/2; -sqrt(2)*a/2 -sqrt(2)*a/2]
# IF Ω₂ = {r : |r| < R, r ∈ ℝ², R ∈ ℝ} (i.e. disks/cylinders)
Rad = 10
V_2 = pi*Rad^2                              #Volume definition required


#Code starts here
Init_Workspace(wl = wl, φ = φ, θ = θ, NG = NG, ϵ_bg = ϵ_bg,
    ϵ_m = mat_file, A = A, Rad = Rad, V_2 = V_2)

# update_dependencies!(NG=20,θ=0,φ=90,wl=300000/814)

#
# ###########################################
#
# wlsweep = 800 : -0.5 : 300
# kmodes = zeros(ComplexF64,(size(wlsweep,1),2))
# curvecs = zeros(ComplexF64,(3,size(wlsweep,1),2))
# QEPmodes = zeros(ComplexF64,(size(wlsweep,1),2))
# sorttol = 1e-2
# for (nl,λ) in enumerate(wlsweep)
#     update_dependencies!(wl=λ)
#     println(p.lambda)
#     kmodes[nl, :], curvecs[:,nl,:], QEPmodes[nl,:] = getMwQEP()
#     if nl != 1 && abs(QEPmodes[nl, 1] - QEPmodes[nl-1, 1]) > abs(QEPmodes[nl, 1] - QEPmodes[nl-1, end])
#         QEPmodes[nl, :] = QEPmodes[nl, end:-1:1]
#     end
#     if nl != 1 && abs(kmodes[nl, 1] - kmodes[nl-1, 1]) > abs(kmodes[nl, 1] - kmodes[nl-1, end])
#         kmodes[nl, :] = kmodes[nl, end:-1:1]
#     end
# end
#
# f_v = 3e5 ./ collect(wlsweep)
# bands_path = string(pwd(), "\\Data\\BS_QEP_R10_90-60_.dat")
# open(bands_path, "w") do io
#     write(io, "Frequency Re(k1) Im(k1) Re(k2) Im(k2)\n")
#     writedlm(io, hcat(real.(f_v), real.(QEPmodes[:,1]), imag.(QEPmodes[:,1]), real.(QEPmodes[:,2]), imag(QEPmodes[:,2])))
# end
#
# f_v = 3e5 ./ collect(wlsweep)
# bands_path = string(pwd(), "\\Data\\BS_R10_90-60.dat")
# open(bands_path, "w") do io
#     write(io, "Frequency Re(k1) Im(k1) Re(k2) Im(k2)\n")
#     writedlm(io, hcat(real.(f_v), real.(kmodes[:,1]), imag.(kmodes[:,1]), real.(kmodes[:,2]), imag(kmodes[:,2])))
# end
###########################################
#Maxwell-Garnett
# wlsweep = 800 : -0.5 : 300
# ksMG = zeros(ComplexF64,(size(wlsweep,1),2))
# sorttol = 1e-2
# fill = l.V_2/l.V
# for (nl,λ) in enumerate(wlsweep)
#     update_dependencies!(wl=λ)
#     effTE = fill*p.e_m + (1-fill)*p.e_bg
#     effTM = p.e_bg * ( (1+fill)*p.e_m + (1-fill)*p.e_bg )/( (1-fill)*p.e_m + (1+fill)*p.e_bg )
#     ksMG[nl,1] = sqrt(effTE)*p.k_0
#     ksMG[nl,2] = sqrt(effTM)*p.k_0
# end
#
# f_v = 3e5 ./ collect(wlsweep)
# bands_path = string(pwd(), "\\Data\\bandstrct_MG_R10.dat")
# open(bands_path, "w") do io
#     writedlm(io, hcat(real.(f_v), real.(ksMG[:,1]), imag.(ksMG[:,1]), real.(ksMG[:,2]), imag(ksMG[:,2])))
# end


# ##########################################
# update_dependencies!(wl=700,NG=30,θ=0,φ=90)
#
# inclsweep = 0 : 0.5 : 90
# kmodes = zeros(ComplexF64,(size(inclsweep,1),2))
# curvecs = zeros(ComplexF64,(3,size(inclsweep,1),2))
# sorttol = 1e-2
# for (nt,th) in enumerate(inclsweep)
#     update_dependencies!(θ=th)
#     kmodes[nt,:], curvecs[:,nt,:] = getMode()
#     if nt != 1 && abs(kmodes[nt, 1] - kmodes[nt-1, 1]) > abs(kmodes[nt, 1] - kmodes[nt-1, end])
#         kmodes[nt, :] = kmodes[nt, end:-1:1]
#     end
# end
#
# incls = collect(inclsweep)
# angles_path = string(pwd(), "\\Data\\anglesweep_700nm_azim90.dat")
# open(angles_path, "w") do io
#            writedlm(io, hcat(real.(incls), real.(kmodes[:,1]), imag.(kmodes[:,1]), real.(kmodes[:,2]), imag(kmodes[:,2])))
# end
# ###########################################

##########################################
# update_dependencies!(wl=700,NG=100,θ=0,φ=0)
#
# aoisweep = 0 : 0.5 : 90
# azimsweep = 0 : 0.5 : 90
# kmodes = zeros(ComplexF64,(size(aoisweep,1),size(azimsweep,1),2))
# curvecs = zeros(ComplexF64,(3,size(aoisweep,1),size(azimsweep,1),2))
# sorttol = 1e-2
# for (np,ph) in enumerate(azimsweep)
#     println(np)
#     update_dependencies!(φ=ph)
#         for (nt,th) in enumerate(azimsweep)
#             update_dependencies!(θ=th)
#             kmodes[nt,np,:], curvecs[:,nt,np,:] = getMode()
#             if nt != 1 && abs(kmodes[nt,np, 1] - kmodes[nt-1,np, 1]) > abs(kmodes[nt,np, 1] - kmodes[nt-1,np, end])
#                 kmodes[nt,np, :] = kmodes[nt,np, end:-1:1]
#             end
#         end
# end
#
# angleHM = [[(ph-1)/2,(th-1)/2,real(kmodes[th,ph,1]),imag(kmodes[th,ph,1]),real(kmodes[th,ph,2]),imag(kmodes[th,ph,2])] for ph in 1:181 for th in 1:181]
#
# angles_path = string(pwd(), "\\Data\\2Daoi_corrected.dat")
# open(angles_path, "w") do io
#     write(io, "phi theta Re(k1) Im(k1) Re(k2) Im(k2)\n")
#     writedlm(io, angleHM)
# end
###########################################


update_dependencies!(NG=50)

ksols,csols = getMode()

res = 0.25
yrange = 2*sqrt(2)*a
zrange = 2*sqrt(2)*a
mode = 2
E = getE_Field(ksols[mode], csols[:,mode], yrange, zrange, res)
E_x = E[1,:,:]
E_y = E[2,:,:]
E_z = E[3,:,:]
# ReNorm = maximum(hcat(abs.(real.(E[1,:,:])),
# abs.(real.(E[2,:,:])),
# abs.(real.(E[3,:,:]))))

E_I =  dropdims(sum(abs.(E).^2,dims=1),dims=1)
pltnrm = maximum(E_I)

E_plot = E_I/pltnrm
# E_plot = real.(E_z)/pltnrm
heatmap(E_plot)

heatmp = [ [(y-1)*res-yrange/2,(z-1)*res,
real.(E_x[z,y]),imag.(E_x[z,y]),
real.(E_y[z,y]),imag.(E_y[z,y]),
real.(E_z[z,y]),imag.(E_z[z,y])] for
y in 1:size(E_plot)[2] for
z in 1:size(E_plot)[1] ]

file_path = string(pwd(), "\\Data\\HM_",trunc(Int, p.lambda),"nm_TM_sq11_0-60.dat")

open(file_path, "w") do io
    write(io, "y z Re(Ex) Im(Ex) Re(Ey) Im(Ey) Re(Ez) Im(Ez)\n")
    writedlm(io, heatmp)
end

# #####################################
# #One Wigner-Seitz
# res = 0.25
# yrange = 3*a/2
# zrange = sqrt(3)*a/2
# mode = 2
# E = getE_Field(ksols[mode], csols[:,mode], yrange, zrange, res)
# E_x = E[1,:,:]
# E_y = E[2,:,:]
# E_z = E[3,:,:]
# ReNorm = maximum(hcat(abs.(real.(E[1,:,:])),
# abs.(real.(E[2,:,:])),
# abs.(real.(E[3,:,:]))))
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
