using Distributed, BenchmarkTools, Plots, Plots.PlotMeasures, LinearAlgebra, DelimitedFiles, ColorSchemes, Dates, ProgressBars
rmprocs(2:1000)
addprocs(1)

@everywhere using Pkg
@everywhere Pkg.activate("./Code/MMSolver")

@everywhere using MMSolver

freq = 400
λ = 2.99792458e5/freq      #wavelength in nm
φ = 90      #azimuthal angle of incidence, do not change in 1D for fixed y-z plane of incidence
θ = 0       #polar angle of incidence
ϵ_bg = 1 + 0im  #permittivity of background medium
mat_file = "Ag_JC_nk.txt"   #file storing permittivities of medium in sphere. Format as in refractiveindex.info files
a = 30.0    #lattice constant
A = [a/2 a; sqrt(3)*a/2 0]  #real space lattice matrix (see Lattice struct in parameters.jl)
Rad = 10.0  #radius of the d-sphere
NG = 1600

Nth = 201
Nr = 101
yz = zeros((2,Nr,Nth))
for (nr,r) in enumerate(range(0,stop=Rad-1e-5,length=Nr))
    for (nth,th) in enumerate(range(0,stop=2*pi,length=Nth))
        yz[1,nr,nth] = 3/4*a + r*cos(th)
        yz[2,nr,nth] = sqrt(3)/4*a + r*sin(th)
    end
end
writedlm("positions.txt",reshape(yz,(2,Nr*Nth))')
