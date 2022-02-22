using Distributed, BenchmarkTools, Plots, Plots.Measures, LinearAlgebra, DelimitedFiles, ColorSchemes, Dates, ProgressBars
rmprocs(2:1000)
addprocs(1)

@everywhere using Pkg
@everywhere Pkg.activate("./Code/MMSolver")

@everywhere using MMSolver

#include("heatmap.jl") #For data (e.g. contour lines)

# freq_sweep = ProgressBar(375:1:1000)
#     lattice,parameters = init_workspace(
#                         λ = 2.99792458e5/freq,                  #wavelength in nm
#                         φ = 90, θ = 0,                          #azimuthal and polar angle of incidence
#                         NG = 1600,                                #reciprocal lattice cut-off (see Lattice struct in parameters.jl)
#                         ϵ_1 = 1+0im,                            #permittivity of background medium
#                         ϵ_2 = "Ag_JC_nk.txt",                   #file storing permittivities of medium in sphere. Format as in refractiveindex.info files
#                         A = [30.0/2 30.0; sqrt(3)*30.0/2 0],    #real space lattice matrix (see Lattice struct in parameters.jl)
#                         Rad = 10.0,                             #radius of the d-sphere
#                         polydegs = (4,4))                       #polynomial degree of basis functions for driving current in Ω_2

        ########################################################################
        # Bandstructure
        ### needs if-statement for init_k
        # if freq ==
        #     fillf = lattice.V_2/lattice.V
        #     init_k = parameters.k_0 * sqrt(((1-fillf)*parameters.eps_1 + (1+fillf)*parameters.eps_2)/((1+fillf)*parameters.eps_1 + (1-fillf)*parameters.eps_2))
        # end
        ### Write header in output-file to record parameters ###
        #     open(bands_path, "a") do io
        #         write(io, string("NG = ",lattice.NG,", Rad = ", lattice.R,", (φ, θ) = (",parameters.azim,",", parameters.polar,"), polydegs = ",parameters.polydegs, ", material: ",parameters.material,"\n"))
        #     end

        # local kmode,evec = get_single_mode(lattice,parameters;manual_ks=init_k)
        # global init_k = kmode

        # open(bands_path, "a") do io
        #     writedlm(io, hcat(real.(radius), real.(kmode), imag.(kmode)))
        # end


        # set_multiline_postfix(rads, "Solution for $freq THz: $kmode          ")
# end

#########################################################################
### Field/Current Plot Data ###
# kmode,evec = get_single_mode(lattice,parameters;manual_ks=0.045586683318467554+0.0036615182514062963im)

freqs = [400,838,900]
kmodes = [0.013270878853772414+7.042605190951711e-6im 0.0005606886180370765+0.2612128351596212im;0im 0im;
        -0.04107536719185247+0.1548678159144141im 0.006578321824730451+0.013109898776823451im]

freq = 400
mode = 1
kmode = kmodes[findfirst(isequal(freq),freqs),mode]
evec = readdlm(string(pwd(),"\\Results\\evec_NG1600_",freq,"THz_TM",mode,".txt"),'\n',ComplexF64)

lat,param = init_workspace(
                        λ = 2.99792458e5/freq,                  #wavelength in nm
                        φ = 90, θ = 0,                          #azimuthal and polar angle of incidence
                        NG = 50,                                #reciprocal lattice cut-off (see Lattice struct in parameters.jl)
                        ϵ_1 = 1+0im,                            #permittivity of background medium
                        ϵ_2 = "Ag_JC_nk.txt",                   #file storing permittivities of medium in sphere. Format as in refractiveindex.info files
                        A = [30.0/2 30.0; sqrt(3)*30.0/2 0],    #real space lattice matrix (see Lattice struct in parameters.jl)
                        Rad = 10.0,                             #radius of the d-sphere
                        polydegs = (2,2))                       #polynomial degree of basis functions for driving current in Ω_2

println("Starting Current calculation for ",freq," THz @ ",Dates.format(Dates.now(),"HH:MM"))

Nth = 201
Nr = 101
C = MMSolver.getC_current_grid(lat,param,kmode,evec; Nth=Nth, Nr=Nr)
Cabs = sqrt.(abs.(C[1,:,:]).^2 + abs.(C[2,:,:]).^2 + abs.(C[3,:,:]).^2)
Cmax = maximum(Cabs)
ths = range(0,stop=2*pi,length=Nth)
rs  = range(0,stop=lat.R,length=Nr)
heatmap(ths,rs,
            Cabs/Cmax,aspect_ratio=:equal,projection=:polar,color=:hot,
                interpolate=true,right_margin=5mm,bottom_margin=5mm,
                axis=false,yticks=[],yrange=(0,10.1))
plot!(t->t,t->10,0,2π,color=:silver,legend=false,linewidth=4)

Nth = 25
th0s = range(0,stop=2*pi,length=Nth)
Nn = 1000
del = 100/Nn
rs = zeros((Nth,Nn))
ths = zeros((Nth,Nn))
pm = ones(Int,(Nth))
Nns = zeros(Int,(Nth))
Qq, Pp0, deg_list = MMSolver.getQq(param.polydegs)
k_v = [param.k_x, param.k_y, kmode]
for (nth,th) in enumerate(th0s)
    rs[nth,1]  = lat.R
    ths[nth,1] = th
    y = lat.R*cos(th)
    z = lat.R*sin(th)
    Cpos = MMSolver.getC_current_pos(lat,param,evec,deg_list,k_v,y,z)
    Cpos = del*real(Cpos)/ sqrt(real(Cpos[2])^2 + real(Cpos[3])^2)
    if (y+Cpos[2])^2+(z+Cpos[3])^2 > lat.R^2
        pm[nth] = -1
    end
    y = y + pm[nth]*Cpos[2]
    z = z + pm[nth]*Cpos[3]
    rs[nth,2] = sqrt(y^2+z^2)
    ths[nth,2] = atan(z,y)
    for nn in 3:Nn
        Cpos = MMSolver.getC_current_pos(lat,param,evec,deg_list,k_v,y,z)
        Cpos = del*real(Cpos)/ sqrt(real(Cpos[2])^2 + real(Cpos[3])^2)
        y = y + pm[nth]*Cpos[2]
        z = z + pm[nth]*Cpos[3]
        rs[nth,nn] = sqrt(y^2+z^2)
        ths[nth,nn] = atan(z,y)
        if rs[nth,nn] > lat.R
            Nns[nth] = nn
            break
        end
    end
end
for nth in 1:Nth
    plot!(ths[nth,1:Nns[nth]],rs[nth,1:Nns[nth]],linewidth=3,color=:silver)
end


#while true
#end


#nrs  = [1,20,35,47,56,65,75,85]
#nths = 1:10:200
#nrs  = [nr  for nr in nrs for nth in nths]
#nths = [nth for nr in nrs for nth in nths]
#Cth = zeros(length(nrs))
#Cr  = zeros(length(nrs))
#for nn in 1:length(nrs)
#    nr = nrs[nn]
#    nth= nths[nn]
#    r = rs[nr]
#    th= ths[nth]
#    Cy = real(C[2,nr,nth])/Cabs[nr,nth]
#    Cz = real(C[3,nr,nth])/Cabs[nr,nth]
#    Cth[nn] = atan(Cz+r*sin(th),Cy+r*cos(th)) - th
#    Cr[nn]  = sqrt((Cy+r*cos(th))^2 + (Cz+r*sin(th))^2) - r
#end
#us = [1,1,1,1]
#vs = [0,0,0,0]
#quiver!(ths[nths],rs[nrs],quiver=(Cth,Cr),color=:silver,linewidth=0.4)

savefig(string(pwd(),"\\Results\\Current_NG1600pd",param.polydegs,"_TMk",mode,"_",freq,".png"))
### print raw plot data ###
# data_path_plot = string(pwd(),"\\Results\\Current-pol_pd44_",freq,"_TM",mode,".txt")
# open(data_path_plot, "w") do io
#     writedlm(io, Cur)
# end


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
