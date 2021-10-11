using Distributed, BenchmarkTools, Plots, LinearAlgebra, DelimitedFiles, Dates
rmprocs(2:1000)
addprocs(0)

@everywhere using Pkg
@everywhere Pkg.activate("./Code/MMSolver")

@everywhere using MMSolver

#For experimentation
include("heatmap.jl")

#Parameters set by the user (lengths in nm, angles in degrees)
for freq in 820:2:890
# freq = 700
    println(freq," @ ",Dates.format(now(), "HHhMM"))
    λ = 2.99792458e5/freq      #wavelength in nm
    φ = 90      #azimuthal angle of incidence, do not change in 1D for fixed y-z plane of incidence
    θ = 0       #polar angle of incidence
    NG = 50    #reciprocal lattice cut-off (see Lattice struct in parameters.jl)
    ϵ_bg = 1 + 0im  #permittivity of background medium
    mat_file = "Ag_JC_nk.txt"   #file storing permittivities of medium in sphere. Format as in refractiveindex.info files
    a = 30.0    #lattice constant
    A = [a/2 a; sqrt(3)*a/2 0]  #real space lattice matrix (see Lattice struct in parameters.jl)
    Rad = 10.0  #radius of the d-sphere
    polydegs=(2,2)

    lattice,parameters = init_workspace(λ = λ, φ = φ, θ = θ, NG = NG, ϵ_1 = ϵ_bg,
                        ϵ_2 = mat_file, A = A, Rad = Rad)

    REbounds = [-2*pi/(sqrt(3)*30),2*pi/(sqrt(3)*30)] #Brillouin Zone: +/- 2*pi/(sqrt(3)*30)
    IMbounds = [0,0.1]
    REheatres, IMheatres = 200, 100
    RErange = LinRange(REbounds[1],REbounds[2],REheatres)
    IMrange = LinRange(IMbounds[1],IMbounds[2],IMheatres)

    real_cont, imag_cont = det_contours(RErange, IMrange, polydegs, lattice, parameters)
    p1 = contour(RErange,IMrange,real_cont,levels=[0],fill=false,c=:red)
    contour!(RErange,IMrange,imag_cont,levels=[0],fill=false,c=:blue)
    plot(p1,title = string("contoursReIm(det(M)) ",2.99792458e5/parameters.lambda))
    savefig(string(pwd(),"\\Results\\TEST_contours_",freq,".png"))
    data_path_real = string(pwd(),"\\Results\\TEST_contours_",freq,"_real.txt")
    data_path_imag = string(pwd(),"\\Results\\TEST_contours_",freq,"_imag.txt")
    open(data_path_real, "w") do io
        writedlm(io, real_cont)
    end
    open(data_path_imag, "w") do io
        writedlm(io, imag_cont)
    end


# savefig(det_map_imag,string(pwd(),"\\Results\\imagdetM_heatmap_",freq,".png"))
# ks, cs = get_polyx_mode(polydegs,lattice,parameters;manual_ks=[0.0013075499639302794+0.011277911210090318im,0.023+0.0372im])
# println(ks)
# evals_Mk,evecs_Mk  = eigenvalues_Mk(polydegs,0.0175+0.0325im,lattice,parameters)
# println(evals_Mk)

end
