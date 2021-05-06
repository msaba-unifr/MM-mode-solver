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
NG = 10
ϵ_bg = 1 + 0im
mat_file = "Ag_JC_nk.txt"
a = 30.0                            #lattice constant
# IF Ω₂ = {r : |r| < R, r ∈ ℝ², R ∈ ℝ} (i.e. disks/cylinders)
Rad = 10.0
V_2 = 2*Rad                              #Volume definition required
mmdim = 1

#Code starts here
Init_Workspace(wl = wl, φ = φ, θ = θ, NG = NG, ϵ_bg = ϵ_bg,
    ϵ_m = mat_file, A = a, Rad = Rad, V_2 = V_2, mmdim = mmdim)

o_vec = zeros(ComplexF64, (3,1))
𝓗invs = getHinv(Gs,o_vec, p.k_1)

ksCuEP,csCuEP = getCuEP(2,𝓗invs, p.k_1, p.k_2, p.k_x, p.k_y,l.V_2, l.V)

ksolspoly4,csolspoly4 = getpoly4Mode()
ksolspolyx4,csolspolyx4 = getpolyxMode(0)

lam_ana,v_ana = solve_analytical(p,l,0)

Nz = 100
QEP_field = [getD2field_poly4(ksQEPpoly4[16],csQEPpoly4[:,16],l,z) for z in LinRange(-l.V_2/2,l.V_2/2,Nz)]
NLEVP_field = [getD2field_poly4(ksolspoly4[1],csolspoly4[:,1],l,z) for z in LinRange(-l.V_2/2,l.V_2/2,Nz)]
ana_field = [getD2field_ana(lam_ana,v_ana,p,l,z) for z in LinRange(-l.V_2/2,l.V_2/2,Nz)]

println()
k_ana = log.(lam_ana[1])/1im/a
println("kQEPpoly2,kpoly2 = ")
println(ksQEPpoly2[10])
println(ksolspoly2[1])
println("kQEPpoly4,kpoly4 = ")
println(ksQEPpoly4[16])
println(ksolspoly4[1])
println("kana = ")
println(k_ana)
print("eps_kQEPpoly2 = ")
println(abs(1 - ksQEPpoly2[10]/k_ana))
print("eps_kQEPpoly4 = ")
println(abs(1 - ksQEPpoly4[16]/k_ana))
print("eps_kNLEPpoly2 = ")
println(abs(1 - ksolspoly2[1]/k_ana))
print("eps_kNLEPpoly4 = ")
println(abs(1 - ksolspoly4[1]/k_ana))
println("eps_fieldQEP,eps_fieldNLEVP = ")
println(norm((1 .- QEP_field./ana_field))/sqrt(Nz))
println(norm((1 .- NLEVP_field./ana_field))/sqrt(Nz))

# E = [getEfield_9D(ksols[1],csols[:,1],l,p,z) for z in LinRange(-l.V_2/2,l.V_2/2,Nz)]/getEfield_9D(ksols[1],csols[:,1],l,p,0)


# E = getE_Field(ksols[2], csols[:,2], 2*a, sqrt(3)*a, 0.25)
#
#
# E_I =  dropdims(sum(abs.(E).^2,dims=1),dims=1)
# E_I = E_I/maximum(E_I)
# heatmap(E_I)
