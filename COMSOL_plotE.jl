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
mode = "TM2"
normEfile = string("normE_",mode,"_",freq,"THz.txt")

COMSOL_E = readdlm(normEfile,comments=true,comment_char='%')
maxE = maximum(COMSOL_E[:,3])
normE = reshape(COMSOL_E[:,3],(Nr,Nth))/maxE
heatmap(range(0,stop=2*pi,length=Nth),range(0,stop=Rad,length=Nr),
    normE,aspect_ratio=:equal,projection=:polar,color=:hot,
        interpolate=true,right_margin=5mm,bottom_margin=5mm,
        axis=false,yticks=[],yrange=(0,10.1))
plot!(t->t,t->10,0,2π,color=:silver,legend=false,linewidth=2)
savefig(string(pwd(),"\\Results\\COMSOL_",mode,"_",freq,"THz.png"))
