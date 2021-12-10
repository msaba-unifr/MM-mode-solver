using Distributed, BenchmarkTools, Plots, Plots.PlotMeasures, LinearAlgebra, DelimitedFiles, ColorSchemes, Dates, ProgressBars
rmprocs(2:1000)
addprocs(4)

@everywhere using Pkg
@everywhere Pkg.activate("./Code/MMSolver")

@everywhere using MMSolver

#include("heatmap.jl") #For data (e.g. contour lines)

# rads = ProgressBar(9:-0.01:8.5)
# freq_sweep = ProgressBar(850:0.1:852)
# for freq in freq_sweep
### Output ###
    # bands_path = string(pwd(), "\\Results\\Ag_Rad9to85_TM2_",freq,".dat")
    # open(bands_path, "w") do io
    #     write(io, string(now(),"\nFrequency Re(k) Im(k)\n"))
    # end
freq = 900
lattice,parameters = init_workspace(
                        λ = 2.99792458e5/freq,                  #wavelength in nm
                        φ = 90, θ = 0,                          #azimuthal and polar angle of incidence
                        NG = 1600,                                #reciprocal lattice cut-off (see Lattice struct in parameters.jl)
                        ϵ_1 = 1+0im,                            #permittivity of background medium
                        ϵ_2 = "Ag_JC_nk.txt",                   #file storing permittivities of medium in sphere. Format as in refractiveindex.info files
                        A = [30.0/2 30.0; sqrt(3)*30.0/2 0],    #real space lattice matrix (see Lattice struct in parameters.jl)
                        Rad = 10.0,                             #radius of the d-sphere
                        polydegs = (4,4))                       #polynomial degree of basis functions for driving current in Ω_2
### NG=800, polydegs=(4,3)
        ########################################################################
        # Bandstructure
            ### needs if-statement for init_k
            ### Write header in output-file to record parameters ###
        #     open(bands_path, "a") do io
        #         write(io, string("NG = ",lattice.NG,", Rad = ", lattice.R,", (φ, θ) = (",parameters.azim,",", parameters.polar,"), polydegs = ",parameters.polydegs, ", material: ",parameters.material,"\n"))
        #     end

init_k = 4*pi/sqrt(3)/30.0-0.04107536719185247+0.1548678159144141im
kmode,evec = get_single_mode(lattice,parameters;manual_ks=init_k)
        # global init_k = kmode

        ### MG ###
        # fillf = lattice.V_2/lattice.V
        # kmode =  parameters.k_0 * sqrt(((1-fillf)*parameters.eps_1 + (1+fillf)*parameters.eps_2)/((1+fillf)*parameters.eps_1 + (1-fillf)*parameters.eps_2))

        # open(bands_path, "a") do io
        #     writedlm(io, hcat(real.(radius), real.(kmode), imag.(kmode)))
        # end


        # set_multiline_postfix(rads, "Solution for $freq THz: $kmode          ")

        #########################################################################
### Field Plot Data ###
# kmode,evec = get_single_mode(lattice,parameters;manual_ks=0.045586683318467554+0.0036615182514062963im)
# local k_label = round(kmode,sigdigits=2)
lattice,parameters = init_workspace(
                        λ = 2.99792458e5/freq,                  #wavelength in nm
                        φ = 90, θ = 0,                          #azimuthal and polar angle of incidence
                        NG = 50,                                #reciprocal lattice cut-off (see Lattice struct in parameters.jl)
                        ϵ_1 = 1+0im,                            #permittivity of background medium
                        ϵ_2 = "Ag_JC_nk.txt",                   #file storing permittivities of medium in sphere. Format as in refractiveindex.info files
                        A = [30.0/2 30.0; sqrt(3)*30.0/2 0],    #real space lattice matrix (see Lattice struct in parameters.jl)
                        Rad = 10.0,                             #radius of the d-sphere
                        polydegs = (4,4))                       #polynomial degree of basis functions for driving current in Ω_2
println("Starting E-Field calculation for ",freq," THz @ ",Dates.format(Dates.now(),"HH:MM"))
field = getE_Field(lattice, parameters, kmode, evec, img_yrange=30.0, img_zrange=0.5*sqrt(3)*30.0, res=0.25)
### print raw e field data ###
data_path_efield = string(pwd(),"\\Results\\E-Field_pd44_",freq,"_TM1.txt")
open(data_path_efield, "w") do io
    writedlm(io, field)
end


###############################################################################
### Contour-Lines ###
# REbounds = [-2*pi/(sqrt(3)*30),2*pi/(sqrt(3)*30)] #Brillouin Zone: +/- 2*pi/(sqrt(3)*30)
# IMbounds = [0,0.1]
# REheatres, IMheatres = 200, 100
# RErange = LinRange(REbounds[1],REbounds[2],REheatres)
# IMrange = LinRange(IMbounds[1],IMbounds[2],IMheatres)
#
# real_cont, imag_cont = det_contours(RErange, IMrange, polydegs, lattice, parameters)
# p1 = contour(RErange,IMrange,real_cont,levels=[0],fill=false,c=:red)
# contour!(RErange,IMrange,imag_cont,levels=[0],fill=false,c=:blue)
# plot(p1,title = string("contoursReIm(det(M)) ",2.99792458e5/parameters.lambda))
# savefig(string(pwd(),"\\Results\\TEST_contours_",freq,".png"))
# data_path_real = string(pwd(),"\\Results\\TEST_contours_",freq,"_real.txt")
# data_path_imag = string(pwd(),"\\Results\\TEST_contours_",freq,"_imag.txt")
# open(data_path_real, "w") do io
#     writedlm(io, real_cont)
# end
# open(data_path_real, "w") do io
#     writedlm(io, imag_cont)
# end

# savefig(det_map_imag,string(pwd(),"\\Results\\imagdetM_heatmap_",freq,".png"))
# ks, cs = get_polyx_mode(polydegs,lattice,parameters;manual_ks=[0.0013075499639302794+0.011277911210090318im,0.023+0.0372im])
# println(ks)
# evals_Mk,evecs_Mk  = eigenvalues_Mk(polydegs,0.0175+0.0325im,lattice,parameters)
# println(evals_Mk)
