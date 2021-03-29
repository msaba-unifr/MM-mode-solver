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
NG = 10
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


update_dependencies!()

ksols,csols = getMode()

ksQEP3D =
ksQEP9D =

E = getE_Field(ksols[2], csols[:,2], 2*a, sqrt(3)*a, 0.25)


E_I =  dropdims(sum(abs.(E).^2,dims=1),dims=1)
E_I = E_I/maximum(E_I)
heatmap(E_I)
