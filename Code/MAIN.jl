using LinearAlgebra
using Einsum
using SpecialFunctions
using NonlinearEigenproblems
using DelimitedFiles
using Interpolations
include("methods.jl")

#Parameters set by the user
wl = 600
φ = 90 #for k_x
θ = 0 #for k_y
NG = 9
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

update_dependencies!(NG = 10)

ksols,csols = getMode()

res = 0.5
yrange = 3*a
zrange = 1.5*sqrt(3)*a
E = getE_Field(ksols[2], csols[:,2], 3*a, 1.5*sqrt(3)*a, res)

E_I =  dropdims(sum(abs.(E).^2,dims=1),dims=1)
E_I = E_I/maximum(E_I)
# using Plots
# heatmap(E_I)

heatmp = [ [(y-1)*res-yrange/2,(z-1)*res-zrange/2,E_I[z,y]] for y in 1:size(E_I)[2] for z in 1:size(E_I)[1] ]

file_path = string(pwd(), "\\Data\\test.dat")

open(file_path, "w") do io
    writedlm(io, heatmp)
end
