using LinearAlgebra
using Einsum
using SpecialFunctions
using DelimitedFiles
using Interpolations
using Plots
using PlyIO
include("methods.jl")

using Profile


#Parameters set by the user
wl = 600
φ = 45 #for k_x
θ = 45 #for k_y
NG = 20
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


bmtimes_faces, error_faces = BM()

plot(10 .^ collect(1:5), bmtimes_faces, xaxis=:log, yaxis=:log, xlabel="faces",
    ylabel="time [s]")
plot(10 .^ collect(1:5), abs.(error_faces), xaxis=:log, yaxis=:log, xlabel="faces",
        ylabel="relative error")

bmtimes_NGs = BM()

plot(2 .^ collect(2:8), bmtimes_NGs, xaxis=:log, yaxis=:log, xlabel="NG",
        ylabel="time[s]")

Gs_test = [[Gs[1,i,j,k], Gs[2,i,j,k], Gs[3,i,j,k]] for i in 1:2*l.NG+1, j in 1:2*l.NG+1, k in 1:1]

t0 = time()
@profiler BM_broadcast(Gs_test)
t1 = time()-t0


t0 = time()
BM_broadcast(Gs_test)
t1 = time()-t0

t0 = time()
BM_listc(Gs_test)
t1 = time()-t0

t0 = time()
for i in 1:10086000
    exp(rand(1)[1])
end
t1 = time()-t0

t0 = time()
@profiler nIP = BM_listc()
t1 = time()-t0

err = sum(IP-nIP)
for i in 1:41, j in 1:41
    println(nIP[i,j], "  |  i= ", i, "  |  j= ", j)
end
update_dependencies!(NG = 20)

ksols,csols = getMode()

E = getE_Field(ksols[2], csols[:,2], 2*a, sqrt(3)*a, 0.25)


E_I =  dropdims(sum(abs.(E).^2,dims=1),dims=1)
E_I = E_I/maximum(E_I)
heatmap(E_I)
