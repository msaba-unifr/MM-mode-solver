using LinearAlgebra
using Einsum
using SpecialFunctions
using DelimitedFiles
using Interpolations
using Plots
include("methods.jl")

#Parameters set by the user
wl = 600
φ = 90
θ = 0
NG = 5
ϵ_bg = 1 + 0im
mat_file = "Ag_JC_nk.txt"
a = 30.0                            #lattice constant
# IF Ω₂ = {r : |r| < R, r ∈ ℝ², R ∈ ℝ} (i.e. disks/cylinders)
Rad = 10.0
V_2 = 2*Rad                              #Volume definition required


#Code starts here
Init_Workspace(wl = wl, φ = φ, θ = θ, NG = NG, ϵ_bg = ϵ_bg,
    ϵ_m = mat_file, A = a, Rad = Rad, V_2 = V_2)


# update_dependencies!(NG=5)
#
# ksols,csols = getMode()

o_vec = zeros(ComplexF64, (3,1))
𝓗invs = getHinv(Gs,o_vec, p.k_1)
ksQEP3D,csQEP3D = getInitGuess(IP²_noDC,𝓗invs, p.k_1, p.k_2, p.k_x, p.k_y,l.V_2, l.V)
ksQEP9D,csQEP9D = getQEP9D(𝓗invs, p.k_1, p.k_2, p.k_x, p.k_y,l.V_2, l.V)

# E = getE_Field(ksols[2], csols[:,2], 2*a, sqrt(3)*a, 0.25)
#
#
# E_I =  dropdims(sum(abs.(E).^2,dims=1),dims=1)
# E_I = E_I/maximum(E_I)
# heatmap(E_I)
