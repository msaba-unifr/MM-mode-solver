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

BM_times = zeros(Float64, (3,8))
powers = 1:8
for power in powers

    t0 = time()
    update_dependencies!(NG = 2^power)
    BM_times[1,power] = time() - t0

    t0 = time()
    ksols,csols = getMode()
    BM_times[2,power] = time() - t0

    t0 = time()
    E = getE_Field(ksols[2], csols[:,2], 2*a, sqrt(3)*a, 0.25)
    BM_times[3,power] = time() - t0

    println("Done: ", 2^power)
end


# E_I =  dropdims(sum(abs.(E).^2,dims=1),dims=1)
# E_I = E_I/maximum(E_I)
# using Plots
# heatmap(E_I)


using Plots
plot([2 4 8 16 32 64 128 256]', BM_times2', xaxis=:log, yaxis=:log, legend=:bottomright)
