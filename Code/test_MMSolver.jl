using Distributed, BenchmarkTools, Plots, Plots.PlotMeasures, LinearAlgebra, DelimitedFiles, ColorSchemes, Dates
rmprocs(2:1000)
addprocs(4)

@everywhere using Pkg
@everywhere Pkg.activate("./Code/MMSolver")

@everywhere using MMSolver

#For experimentation
include("heatmap.jl")
##
# #Parameters set by the user (lengths in nm, angles in degrees)
# bands_path = string(pwd(), "\\Results\\BS_noKappaNG500_TE.dat")
# open(bands_path, "w") do io
#     write(io, string(now(),"\nFrequency Re(k) Im(k)\n"))
# end

# for freq in collect(375:1:1000)
# freq 800 manual_ks=[0.04605052302552293 + 0.003853434757202215im,0.08676257424024099 + 0.042221315595793715im]
# freq 838 manual_ks=[0.009630870180887844 + 0.11854192485740343im,0.019480470421989013 + 0.0571813580703404im]
for freq in [800,838,900]
    # println(freq)
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


    # 820 THz manual_ks=[0.06+0.01im,0.02+0.09im], 844 THz manual_ks=[0.1+0.05im,0.02+0.05im], 880 THz manual_ks=[0.2+0.09im,0.01+0.02im]
    if freq == 800
        init_ks=[0.04605052302552293 + 0.003853434757202215im,0.009630870180887844 + 0.11854192485740343im]
    elseif freq == 838
        init_ks=[0.08676257424024099 + 0.042221315595793715im,0.019480470421989013 + 0.0571813580703404im]
    elseif freq == 900
        init_ks=[0.2535200231544334 + 0.17576182436863946im,0.006514597021692985 + 0.012930216322146883im]
    end
    tmmodes,tmvecs = get_polyx_mode(polydegs,lattice,parameters;manual_ks=init_ks)
    println("Solutions found for ",freq," THz: ",tmmodes)
    ########################################################################
    #Bandstructure-Loop
        # if freq == 375
        #     #TE
        #     init_k = 0+0.025im
        #     #TM1
        #     # init_k = 0.012+0im
        #     #TM2 (from 900 THz)
        #     # init_k = 0.0065145970216929855 + 0.012930216322146876im
        #     #TM2 (from 375 THz)
        #     # init_k = 0.0006435266557436068 + 0.26924804806917046im
        # end
        # kmodes,evecs = get_single_mode(polydegs,lattice,parameters;manual_ks=init_k)
        # println("Solutions for ",freq," THz: ",kmodes)
        #
        # open(bands_path, "a") do io
        #     writedlm(io, hcat(real.(freq), real.(kmodes), imag.(kmodes)))
        # end
        # global init_k = kmodes


    img_yrange = a
    img_zrange = 0.5*sqrt(3)*a
    res = 0.25
    for mode in [1,2]
        println("Starting E-Field calculation for ",freq," THz, mode ",mode," @ ",Dates.format(Dates.now(),"HH:MM"))
        field = getE_Field(polydegs, lattice, parameters, tmmodes[mode], tmvecs[:,mode], img_yrange, img_zrange, res)
        # print raw e field data
        # data_path_efield = string(pwd(),"\\Results\\test_",freq,"-TMk",mode,".txt")
        # open(data_path_efield, "w") do io
        #     writedlm(io, field)
        # end
    end
end
##############Contour-Lines###########################
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
