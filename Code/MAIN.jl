using LinearAlgebra
using Einsum
using SpecialFunctions
using DelimitedFiles
using Interpolations
using Plots
using PyPlot
using NonlinearEigenproblems
include("methods.jl")

#Parameters set by the user
wl = 370
φ = 90   #do not change in 1D for fixed y-z plane of incidence
θ = 0
NG = 200
ϵ_bg = 1 + 0im
mat_file = "Ag_JC_nk.txt"
a = 30.0                            #lattice constant
# IF Ω₂ = {r : |r| < R, r ∈ ℝ², R ∈ ℝ} (i.e. disks/cylinders)
A = [a/2 a; sqrt(3)*a/2 0]
#A = [a 0; 0 a]
Rad = 10.0
mmdim = 2
if mmdim == 1
    V_2 = 2*Rad                              #Volume definition required
elseif mmdim == 2
    V_2 = pi*Rad^2
end
polydegs = (3,3) # tuple of non-negative integers

#Code starts here
Init_Workspace(wl = wl, φ = φ, θ = θ, NG = NG, ϵ_bg = ϵ_bg,
    ϵ_m = mat_file, A = A, Rad = Rad, V_2 = V_2, mmdim = mmdim)

o_vec = zeros(ComplexF64, (3,1))
𝓗invs = getHinv(Gs,o_vec, p.k_1)

ksolspoly,csolspoly = getpolyxMode(polydegs,oldQEP=false)
ksols,csols = getMode()

ksQEP2Dpolyx,csQEP2Dpolyx = getQEPpolyx(polydegs, 𝓗invs, p.k_1, p.k_2, p.k_x, p.k_y, l.V_2, l.V)
ksQEP_old,csQEP_old = getInitGuess(IP²_noDC, 𝓗invs, p.k_1, p.k_2, p.k_x, p.k_y, l.V_2, l.V)

println()
println(polydegs)
println(det(getpolyxM(polydegs,ksolspoly[2])))
println(ksols[2])
println(ksolspoly[2])

eigs = eigen(getpolyxM(polydegs,ksolspoly[2]),sortby=x->abs(x))
println()
println(abs.(eigs.values))
for i in 0:convert(Int,length(eigs.values)/3)-1
    println(abs.(eigs.vectors[1+3*i:3+3*i,1]))
end

#pyplot()
#res = 100
#r = LinRange(0,1,res)
#θ = LinRange(0,360,res)
#field(r,θ) = r^2
#(0.65+0.15*(r*cos.(θ)).^2+0.2*r*sin.(θ)+0.4*y^2*z r.^3*cos.(θ).^2*  +0.12*z^2+0.33*y^2*z^2)^2+(0.14*y+0.1*y*z+0.45*y*z^2)^2
#heatmap(field.(r,θ),aspect_ratio=:equal,proj=:polar,legend=false)

#REheatres = 20
#IMheatres = 10
#REbounds = [0.016,0.018]
#IMbounds = [0,0.0001]
#RErange = LinRange(REbounds[1],REbounds[2],REheatres)
#IMrange = LinRange(IMbounds[1],IMbounds[2],IMheatres)
#heatMDet = [log(abs(det(getpolyxM(polydegs,x+y*im)))) for y in IMrange, x in RErange]
#heatmap(RErange,IMrange,heatMDet)

# ksolspoly4,csolspoly4 = getpolyxMode(4)
# ksolspoly2,csolspoly2 = getpolyxMode(2)
# ksolspolyx0,csolspolyx0 = getpolyxMode(0)
# ksols,csols = getMode()
#
# lam_ana,v_ana = solve_analytical(p,l,0)
# k_ana = log.(lam_ana[1])/1im/a
#
# Nz = 100
# QEP_field = [getD2field_poly2(ksQEP[16],csQEP[:,16],l,z) for z in LinRange(-l.V_2/2,l.V_2/2,Nz)]
# NLEVP2_field = [getD2field_poly2(ksolspoly2[2],csolspoly2[:,2],l,z) for z in LinRange(-l.V_2/2,l.V_2/2,Nz)]
# NLEVP4_field = [getD2field_poly4(ksolspoly4[2],csolspoly4[:,2],l,z) for z in LinRange(-l.V_2/2,l.V_2/2,Nz)]
# ana_field = [getD2field_ana(lam_ana,v_ana,p,l,z) for z in LinRange(-l.V_2/2,l.V_2/2,Nz)]

# println()
# println("kQEPpoly4,kpoly4 = ")
# println(ksQEP[16])
# println(ksolspoly4[1])
# println("kana = ")
# println(k_ana)
# print("eps_k_QEP = ")
# println(abs(1 - ksQEP[16]/k_ana))
# print("eps_k_kNLEPpoly2 = ")
# println(abs(1 - ksolspoly2[2]/k_ana))
# print("eps_k_NLEPpoly4 = ")
# println(abs(1 - ksolspoly4[2]/k_ana))
# print("eps_field_QEP = ")
# println(norm((1 .- QEP_field./ana_field))/sqrt(Nz))
# print("eps_field_kNLEPpoly2 = ")
# println(norm((1 .- NLEVP2_field./ana_field))/sqrt(Nz))
# print("eps_field_NLEPpoly4 = ")
# println(norm((1 .- NLEVP4_field./ana_field))/sqrt(Nz))

# E = [getEfield_9D(ksols[1],csols[:,1],l,p,z) for z in LinRange(-l.V_2/2,l.V_2/2,Nz)]/getEfield_9D(ksols[1],csols[:,1],l,p,0)


# E = getE_Field(ksols[2], csols[:,2], 2*a, sqrt(3)*a, 0.25)
#
#
# E_I =  dropdims(sum(abs.(E).^2,dims=1),dims=1)
# E_I = E_I/maximum(E_I)
# heatmap(E_I)
