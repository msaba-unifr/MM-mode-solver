using LinearAlgebra
using Einsum
using SpecialFunctions
using NonlinearEigenproblems
using DelimitedFiles
using Interpolations
include("functions.jl")


#Parameters set by the user
φ = 0 #for k_x
θ = 60 #for k_y
#ϵ_m = (0.06 + 4.152im)^2
NG = 10
ϵ_bg = 1 + 0im

# IF Ω₂ = {r : |r| < R, r ∈ ℝ³, R ∈ ℝ} (i.e. sphere)
R = 10.0
V_2 = pi*R^2                              #Volume definition required
a = 30.0                                    #lattice constant
Δλ = 1                                        #wavelength spacing
#A = [a 0 0; 0 a 0; 0 0 a]                     #cubic lattice 3D
#A = [a 0; 0 2*a]                               #cubic lattice 2D
#A = [a a/2; 0 sqrt(3)*a/2]                    #hexagonal lattice
A = [a/2 a; sqrt(3)*a/2 0]                    #hexagonal lattice
if size(A)[1] == 2
    V_2 = pi*R^2
end
results_file = "Results.txt"
mat_file = "Ag_JC_nk.txt"


#Code starts here
l = Lattice2D(NG,A,V_2)
#Creating G_space
Gs = getGspace()
IP_1 = dropdims(InnerProd.(sqrt.(sum(Gs.*Gs,dims=1))*R,exclude_DC=true),dims=1).^2
IP_2 = dropdims(InnerProd.(sqrt.(sum(Gs.*Gs,dims=1))*R),dims=1).^2

#Preallocating for result arrays
nmode = 2
wl_v = 600.0 : -Δλ : 600
ks = zeros(ComplexF64, (size(wl_v,1),nmode))
cs = zeros(ComplexF64, (3,size(wl_v,1),nmode))
ksols = zeros(ComplexF64, (size(wl_v,1),nmode))
csols = zeros(ComplexF64, (3,size(wl_v,1),nmode))

#Interpolating (n,k) data from txt file
file_loc = string(pwd(), "\\MaterialModels\\", mat_file)
eps_data = readdlm(file_loc, '\t', Float64, '\n')
itp1 = LinearInterpolation(eps_data[:,1], eps_data[:,2])
itp2 = LinearInterpolation(eps_data[:,1], eps_data[:,3])
eps_ms = (itp1.(wl_v ./ 1000) + itp2.(wl_v ./ 1000) * 1im).^2

#Loop for calculating all initial guesses in wl_v
for (n, wl) in enumerate(wl_v)
    ϵ_m = eps_ms[n]
    #debugging values
    #wl = 750
    #ϵ_m = -26.98634762786314+0.3238006183429164im

    global p = Parameters(wl, φ, θ, ϵ_m, ϵ_bg)

    #Creating 𝓗⁻¹ for initial guess
    o_vec = zeros(ComplexF64, (3,1))
    𝓗inv = getHinv(Gs,o_vec)

    #InitialGuess
    eigs_init = getInitGuess(IP_1,𝓗inv)
    global λs,vs = eigs_init.values, eigs_init.vectors

    nλ = 1
    tol = 1e-3
    for (i, λ_val) in enumerate(λs)

        #Eigenvalue filtering, only works for Φ, Θ = 0 yet
        if real(λ_val) <= 0
            continue
        elseif abs(1- (λ_val^2+p.k_x^2+p.k_y^2)/(p.e_bg*p.k_0^2)) < 1e-8
            continue
        end
        ks[n, nλ] = λs[i]
        cs[:, n, nλ] = vs[4:6, i]
        nλ += 1
    end
    #Reorder eigenvalues if imaginary part suffers a value jump
    if n != 1 && abs(imag(ks[n, 1]) - imag(ks[n-1, 1])) > tol
        ks[n, :] = ks[n, end:-1:1]
        cs[:, n, :] = cs[:, n, end:-1:1]
    end

    println(wl)
end

#Solve NLEVP for each non filtered mode
for mode = 1:2
    for (n, wl) in enumerate(wl_v)

        ϵ_m = eps_ms[n]
        global p = Parameters(wl, θ, φ, ϵ_m, ϵ_bg)
        global IP = (p.k_1^2 - p.k_2^2) / l.V_2 / l.V .* IP_2
        k_sol = scalarNewton(ks[n, mode])
        ksols[n, mode] = k_sol
        println("Mode = ", mode, " | λ= ", wl)
    end
end

#Display Results as a plot ν(k) ,ksols = newton output, ks = qep output
using Plots
#f_v = 3e5 ./ collect(wl_v)
#plot([imag.(ksols), real.(ksols)], f_v)
#plot([imag.(ks), real.(ks)], f_v)
#=
#Write Results to txt file
open(results_file, "w") do io
           writedlm(io, hcat(f_v, ksols))
end
=#

#Ploting field intensity
test_idx = 1
test_mode = 1
img_yrange = 200 #nm
img_zrange = 100 #nm
res = 0.5 #nm

ys = -img_yrange/2 : res : img_yrange/2
zs = -img_zrange/2 : res : img_zrange/2
ksol = ksols[test_idx, test_mode]
csol = qr(conj(getMder(ksol,0)), Val(true)).Q[:,end]
kx = p.k_x
ky = p.k_y
kGs = [kx, ky, ksol] .+ Gs
x_0 = 0.0

𝓗inv = getHinv(Gs, [kx, ky, ksol])
IP = sqrt.(IP_2)
@einsum H_c[i,k,n,m] := IP[k,n,m] * 𝓗inv[i,j,k,n,m] * csol[j]
H_c = H_c ./ l.V

E = [norm(sum(H_c[1,:,:,:] .* exp.(-1im*kGs[1,:,:,:]*x_0)))^2 +
    norm(sum(H_c[2,:,:,:] .* exp.(-1im*kGs[2,:,:,:]*y)))^2 +
    norm(sum(H_c[3,:,:,:] .* exp.(-1im*kGs[3,:,:,:]*z)))^2 for
    z in zs, y in ys]
heatmap(real(E))
