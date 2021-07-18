using Printf
using LinearAlgebra
using Einsum
using SpecialFunctions
using DelimitedFiles
using Interpolations
using Dates
#using Plots
#using PyPlot
#using NonlinearEigenproblems

include("parameters.jl")
include("methods.jl")
# include("deprecated.jl")

#Parameters set by the user (lengths in nm, angles in degrees)
λ = 370     #wavelength in nm
φ = 90      #azimuthal angle of incidence, do not change in 1D for fixed y-z plane of incidence
θ = 0       #polar angle of incidence
NG = 300    #reciprocal lattice cut-off (see Lattice struct in parameters.jl)
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
println()
@printf("Starting program with NG = %d.\n",NG)
t0=time()
Init_Workspace(λ = λ, φ = φ, θ = θ, NG = NG, ϵ_1 = ϵ_bg,
    ϵ_2 = mat_file, A = A, Rad = Rad, mmdim = mmdim)
# @printf("Init workspace took %f minutes.\n",(time()-t0)/60); t1=time()
#
# ksols,csols = getpolyxMode((0,0))
# @printf("Polydegs=(0,0) took %f minutes.\n",(time()-t1)/60); t1=time()
#
# ksolspoly,csolspoly = getpolyxMode(polydegs,oldQEP=true)
# @printf("Polydegs=(%d,%d) took %f minutes.\n\n",polydegs[1],polydegs[2],(time()-t1)/60)
#
# @printf("Polydegs = (0,0) solutions: k1 = %f + %f im and k2 = %f + %f im.\n",
#     real(ksols[1]),imag(ksols[1]),real(ksols[2]),imag(ksols[2]))
# @printf("Polydegs = (%d,%d) solutions: k1 = %f + %f im and k2 = %f + %f im.\n",polydegs[1],polydegs[2],
#     real(ksolspoly[1]),imag(ksolspoly[1]),real(ksolspoly[2]),imag(ksolspoly[2]))

wlsweep = 585 : -5 : 300
f_v = 3e5 ./ collect(wlsweep)
bands_path = string(pwd(), "\\Results\\BS_R10_90-0_poly22_test.dat")
open(bands_path, "w") do io
    write(io, "Frequency Re(k1) Im(k1) Re(k2) Im(k2)\n")
end
kmodes = zeros(ComplexF64,(size(wlsweep,1),2))
curvecs = zeros(ComplexF64,(3*(polydegs[1]+1)*(polydegs[2]+1),size(wlsweep,1),2))
sorttol = 1e-2
t1=time()
for (nl,wl) in enumerate(wlsweep)
    Init_Workspace(λ = wl, φ = φ, θ = θ, NG = NG, ϵ_1 = ϵ_bg,
        ϵ_2 = mat_file, A = A, Rad = Rad, mmdim = mmdim)
    println(p.lambda)
    kmodes[nl, :], curvecs[:,nl,:] = getpolyxMode(polydegs,oldQEP=false)
    if nl != 1 && abs(kmodes[nl, 1] - kmodes[nl-1, 1]) > abs(kmodes[nl, 1] - kmodes[nl-1, end])
        kmodes[nl, :] = kmodes[nl, end:-1:1]
    end
    open(bands_path, "a") do io
        writedlm(io, hcat(real.(f_v[nl]), real.(kmodes[nl,1]), imag.(kmodes[nl,1]), real.(kmodes[nl,2]), imag(kmodes[nl,2])))
    end
    @printf("Runtime %f minutes. Time: %s\n",(time()-t1)/60,Dates.format(now(), "HHhMM"));global t1=time()
end




#eigs = eigen(getpolyxM(polydegs,ksolspoly[2]),sortby=x->abs(x))
#println()
#println(abs.(eigs.values))
#for i in 0:convert(Int,length(eigs.values)/3)-1
#    println(abs.(eigs.vectors[1+3*i:3+3*i,1]))
#end

#pyplot()
#res = 100
#r = LinRange(0,1,res)
#θ = LinRange(0,360,res)
#field(r,θ) = r^2
#(0.65+0.15*(r*cos.(θ)).^2+0.2*r*sin.(θ)+0.4*y^2*z r.^3*cos.(θ).^2*  +0.12*z^2+0.33*y^2*z^2)^2+(0.14*y+0.1*y*z+0.45*y*z^2)^2
#heatmap(field.(r,θ),aspect_ratio=:equal,proj=:polar,legend=false)

#REheatres = 20
#IMheatres = 10
#REbounds = [0.016,0.018]
#IMbounds = [0,0.0001]
#RErange = LinRange(REbounds[1],REbounds[2],REheatres)
#IMrange = LinRange(IMbounds[1],IMbounds[2],IMheatres)
#heatMDet = [log(abs(det(getpolyxM(polydegs,x+y*im)))) for y in IMrange, x in RErange]
#heatmap(RErange,IMrange,heatMDet)

# ksolspoly4,csolspoly4 = getpolyxMode(4)
# ksolspoly2,csolspoly2 = getpolyxMode(2)
# ksolspolyx0,csolspolyx0 = getpolyxMode(0)
# ksols,csols = getMode()
#
# lam_ana,v_ana = solve_analytical(p,l,0)
# k_ana = log.(lam_ana[1])/1im/a
#
# Nz = 100
# QEP_field = [getD2field_poly2(ksQEP[16],csQEP[:,16],l,z) for z in LinRange(-l.V_2/2,l.V_2/2,Nz)]
# NLEVP2_field = [getD2field_poly2(ksolspoly2[2],csolspoly2[:,2],l,z) for z in LinRange(-l.V_2/2,l.V_2/2,Nz)]
# NLEVP4_field = [getD2field_poly4(ksolspoly4[2],csolspoly4[:,2],l,z) for z in LinRange(-l.V_2/2,l.V_2/2,Nz)]
# ana_field = [getD2field_ana(lam_ana,v_ana,p,l,z) for z in LinRange(-l.V_2/2,l.V_2/2,Nz)]

# println()
# println("kQEPpoly4,kpoly4 = ")
# println(ksQEP[16])
# println(ksolspoly4[1])
# println("kana = ")
# println(k_ana)
# print("eps_k_QEP = ")
# println(abs(1 - ksQEP[16]/k_ana))
# print("eps_k_kNLEPpoly2 = ")
# println(abs(1 - ksolspoly2[2]/k_ana))
# print("eps_k_NLEPpoly4 = ")
# println(abs(1 - ksolspoly4[2]/k_ana))
# print("eps_field_QEP = ")
# println(norm((1 .- QEP_field./ana_field))/sqrt(Nz))
# print("eps_field_kNLEPpoly2 = ")
# println(norm((1 .- NLEVP2_field./ana_field))/sqrt(Nz))
# print("eps_field_NLEPpoly4 = ")
# println(norm((1 .- NLEVP4_field./ana_field))/sqrt(Nz))

# E = [getEfield_9D(ksols[1],csols[:,1],l,p,z) for z in LinRange(-l.V_2/2,l.V_2/2,Nz)]/getEfield_9D(ksols[1],csols[:,1],l,p,0)


# E = getE_Field(ksols[2], csols[:,2], 2*a, sqrt(3)*a, 0.25)
#
#
# E_I =  dropdims(sum(abs.(E).^2,dims=1),dims=1)
# E_I = E_I/maximum(E_I)
# heatmap(E_I)
