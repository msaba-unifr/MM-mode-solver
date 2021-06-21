using LinearAlgebra
using Einsum
using SpecialFunctions
using DelimitedFiles
using Interpolations
using Plots
using NonlinearEigenproblems
include("methods.jl")

#Parameters set by the user
wl = 600
φ = 90   #do not change in 1D for fixed y-z plane of incidence
θ = 0
NG = 1
ϵ_bg = 1 + 0im
mat_file = "Ag_JC_nk.txt"
a = 30.0                            #lattice constant
# IF Ω₂ = {r : |r| < R, r ∈ ℝ², R ∈ ℝ} (i.e. disks/cylinders)
A = [a/2 a; sqrt(3)*a/2 0]
Rad = 10.0
mmdim = 2
if mmdim == 1
    V_2 = 2*Rad                              #Volume definition required
elseif mmdim == 2
    V_2 = pi*Rad^2
end
polydegs = (0,0) # tuple of non-negative integers

#Code starts here
Init_Workspace(wl = wl, φ = φ, θ = θ, NG = NG, ϵ_bg = ϵ_bg,
    ϵ_m = mat_file, A = A, Rad = Rad, V_2 = V_2, mmdim = mmdim)

o_vec = zeros(ComplexF64, (3,1))
𝓗invs = getHinv(Gs,o_vec, p.k_1)

ksolspoly,csolspoly = getpolyxMode(polydegs)
ksols,csols = getMode()

ks2Dpolyx,cs2Dpolyx = getQEPpolyx(polydegs, 𝓗invs, p.k_1, p.k_2, p.k_x, p.k_y, l.V_2, l.V)
ksQEP_old,csQEP_old = getInitGuess(IP²_noDC, 𝓗invs, p.k_1, p.k_2, p.k_x, p.k_y, l.V_2, l.V)

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
