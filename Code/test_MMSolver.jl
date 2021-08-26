using Distributed, BenchmarkTools
rmprocs(2:1000)
addprocs(2)

@everywhere using Pkg
@everywhere Pkg.activate("./Code/MMSolver")

@everywhere using MMSolver


#Parameters set by the user (lengths in nm, angles in degrees)
freq = 700
λ = 2.99792458e5/freq      #wavelength in nm
φ = 90      #azimuthal angle of incidence, do not change in 1D for fixed y-z plane of incidence
θ = 0       #polar angle of incidence
NG = 10    #reciprocal lattice cut-off (see Lattice struct in parameters.jl)
ϵ_bg = 1 + 0im  #permittivity of background medium
mat_file = "Ag_JC_nk.txt"   #file storing permittivities of medium in sphere. Format as in refractiveindex.info files
a = 30.0    #lattice constant
A = [a/2 a; sqrt(3)*a/2 0]  #real space lattice matrix (see Lattice struct in parameters.jl)
Rad = 10.0  #radius of the d-sphere
polydegs=(2,2)

lattice,parameters = init_workspace(λ = λ, φ = φ, θ = θ, NG = NG, ϵ_1 = ϵ_bg,
                    ϵ_2 = mat_file, A = A, Rad = Rad)
