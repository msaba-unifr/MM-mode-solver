using Printf
using LinearAlgebra
using Einsum
using SpecialFunctions
using DelimitedFiles
using Interpolations
using Dates

include("parameters.jl")
include("methods.jl")
include("depr_vectorized.jl")

#Parameters set by the user (lengths in nm, angles in degrees)
freq = 700
λ = 2.99792458e5/freq      #wavelength in nm
φ = 90      #azimuthal angle of incidence, do not change in 1D for fixed y-z plane of incidence
θ = 0       #polar angle of incidence
NG = 100    #reciprocal lattice cut-off (see Lattice struct in parameters.jl)
ϵ_bg = 1 + 0im  #permittivity of background medium
mat_file = "Ag_JC_nk.txt"   #file storing permittivities of medium in sphere. Format as in refractiveindex.info files
a = 30.0    #lattice constant
# IF Ω₂ = {r : |r| < R, r ∈ ℝ², R ∈ ℝ} (i.e. disks/cylinders)
A = [a/2 a; sqrt(3)*a/2 0]  #real space lattice matrix (see Lattice struct in parameters.jl)
Rad = 10.0  #radius of the d-sphere
mmdim = 2   #dimension d of the lattice
if mmdim == 1
    V_2 = 2*Rad                              #Volume definition required
elseif mmdim == 2
    V_2 = pi*Rad^2
    polydegs = (2,2)    #maximum degrees (N,M) of polynomial to approximate the current in the d-sphere c=
end

#Code starts here

Init_Workspace(λ = λ, φ = φ, θ = θ, NG = NG, ϵ_1 = ϵ_bg,
    ϵ_2 = mat_file, A = A, Rad = Rad, mmdim = mmdim)

init_k = [ 0.0004376494248136995 + 0.019498110004211807im,   0.02883857058875105 + 0.00031740992926030585im] # for 700 THz
kmode = getpolyxMode(polydegs,manual_ks=init_k)[1]

# freq_sweep = 700 : 2 : 900
# speed_of_light = 2.99792458e5
# bands_path = string(pwd(), "\\Results\\BS_new.dat")
# open(bands_path, "a") do io
#     write(io, string(now(),"\nFrequency Re(k1) Im(k1) Re(k2) Im(k2)\n"))
# end
# sorttol = 1e-2
# init_k = [ 0.0004376494248136995 + 0.019498110004211807im,   0.02883857058875105 + 0.00031740992926030585im]
# t1=time()
# for freq in collect(freq_sweep)
#     λ = speed_of_light / freq
#     Init_Workspace(λ = λ, φ = φ, θ = θ, NG = NG, ϵ_1 = ϵ_bg,
#         ϵ_2 = mat_file, A = A, Rad = Rad, mmdim = mmdim)
#     println(p.lambda)
#     kmode = getpolyxMode(polydegs,manual_ks=init_k)[1]
#     open(bands_path, "a") do io
#         writedlm(io, hcat(real.(freq), real.(kmode[1]), imag.(kmode[1]), real.(kmode[2]), imag(kmode[2])))
#     end
#     global init_k = kmode
#     @printf("Runtime for %f nm was %f minutes, finished @ %s\n",λ,(time()-t1)/60,Dates.format(now(), "HHhMM"));global t1=time()
# end
