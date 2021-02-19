using LinearAlgebra
using Einsum
using SpecialFunctions
using NonlinearEigenproblems
using DelimitedFiles
using Interpolations
include("methods.jl")

#Parameters set by the user
wl = 600
φ = 45 #for k_x
θ = 45 #for k_y
NG = 9
ϵ_bg = 1 + 0im
mat_file = "Ag_JC_nk.txt"
a = 30.0                            #lattice constant
A = [a/2 a; sqrt(3)*a/2 0]
# IF Ω₂ = {r : |r| < R, r ∈ ℝ², R ∈ ℝ} (i.e. disks/cylinders)
Rad = 10.0
V_2 = pi*Rad^2                              #Volume definition required


#Code starts here
t0 = time()
Init_Workspace(wl = wl, φ = φ, θ = θ, NG = NG, ϵ_bg = ϵ_bg,
    ϵ_m = mat_file, A = A, Rad = Rad, V_2 = V_2)
t_init = time()-t0

t0 = time()
update_dependencies!(NG = 10)
t_update = time()-t0

t0 = time()
ksols,csols = getMode()
t_getmode = time()-t0

t0 = time()
E = getE_Field(ksols[2], csols[:,2], 2*a, sqrt(3)*a, 0.25)
t_Efield = time()-t0

println("@tensor-performance")
println("@tensor-performance")
println("t_update :",t_update)
println("t_getmode :",t_getmode)
println("t_Efield :",t_Efield)

# E_I =  dropdims(sum(abs.(E).^2,dims=1),dims=1)
# E_I = E_I/maximum(E_I)
# using Plots
# heatmap(E_I)
