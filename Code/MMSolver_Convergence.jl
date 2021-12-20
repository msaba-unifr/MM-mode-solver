using Distributed, BenchmarkTools, LinearAlgebra, DelimitedFiles
rmprocs(2:1000)
addprocs(1)

@everywhere using Pkg
@everywhere Pkg.activate("./Code/MMSolver")

@everywhere using MMSolver

# #Parameters set by the user (lengths in nm, angles in degrees)
#bands_path = string(pwd(), "\\Results\\BS_noKappaNG500_TE.dat")
#open(bands_path, "w") do io
#    write(io, string(now(),"\nFrequency Re(k) Im(k)\n"))
#end

freq = 838
λ = 2.99792458e5/freq      #wavelength in nm
φ = 90      #azimuthal angle of incidence, do not change in 1D for fixed y-z plane of incidence
θ = 0       #polar angle of incidence
ϵ_bg = 1 + 0im  #permittivity of background medium
mat_file = "Ag_JC_nk.txt"   #file storing permittivities of medium in sphere. Format as in refractiveindex.info files
a = 30.0    #lattice constant
A = [a/2 a; sqrt(3)*a/2 0]  #real space lattice matrix (see Lattice struct in parameters.jl)
Rad = 10.0  #radius of the d-sphere

init_k = 0.084825259818829 + 0.04014433953563797im

for NG in [200,400,800,1600]
    for pdy in [2,4]
        for pdz in [2,3,4]
            lat,param = init_workspace(λ = λ, φ = φ, θ = θ, NG = NG, ϵ_1 = ϵ_bg,
                    ϵ_2 = mat_file, A = A, Rad = Rad,polydegs=(pdy,pdz))
            ksol,csol,iters = get_single_mode(lat,param;kinit=init_k)
            println()
            println("Solutions for NG=",NG," and polydegs=(",pdy,",",pdz,"): ",ksol," (iters=",iters,")")
        end
    end
end
    #open(bands_path, "a") do io
    #    writedlm(io, hcat(real.(freq), real.(kmodes), imag.(kmodes)))
    #end
