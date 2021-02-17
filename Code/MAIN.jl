using LinearAlgebra
using Einsum
using SpecialFunctions
using NonlinearEigenproblems
using DelimitedFiles
using Interpolations
include("functions.jl")


#Parameters set by the user
φ = 45 #for k_x
θ = 45 #for k_y
#ϵ_m = (0.06 + 4.152im)^2
NG = 10
ϵ_bg = 1 + 0im

# IF Ω₂ = {r : |r| < R, r ∈ ℝ², R ∈ ℝ} (i.e. cylinders)
Rad = 10.0
V_2 = pi*Rad^2                              #Volume definition required
a = 30.0                                    #lattice constant
Δλ = 1                                        #wavelength spacing
#A = [a 0 0; 0 a 0; 0 0 a]                     #cubic lattice 3D
#A = [a 0; 0 2*a]                               #cubic lattice 2D
#A = [a a/2; 0 sqrt(3)*a/2]                    #hexagonal lattice
A = [a/2 a; sqrt(3)*a/2 0]                    #hexagonal lattice
if size(A)[1] == 2
    V_2 = pi*Rad^2
end
results_file = "Results.txt"
mat_file = "Ag_JC_nk.txt"


#Code starts here
l = Lattice2D(NG,A,V_2)
#Creating G_space
Gs = getGspace()
IP²_noDC = dropdims(InnerProd.(sqrt.(sum(Gs.*Gs,dims=1))*Rad,exclude_DC=true),dims=1).^2
IP² = dropdims(InnerProd.(sqrt.(sum(Gs.*Gs,dims=1))*Rad),dims=1).^2
IP = dropdims(InnerProd.(sqrt.(sum(Gs.*Gs,dims=1))*Rad),dims=1)
#Preallocating for result arrays
nmode = 2
wl_v = 700.0 : -Δλ : 350
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
    global p = Parameters(wl, φ, θ, ϵ_m, ϵ_bg)

    #Creating 𝓗⁻¹ for initial guess
    o_vec = zeros(ComplexF64, (3,1))
    𝓗invs = getHinv(Gs,o_vec)

    #InitialGuess
    eigs_init = getInitGuess(IP²_noDC,𝓗invs)
    global λs,vs = eigs_init.values, eigs_init.vectors

    nλ = 1
    tol = 1e-3
    for (i, λ_val) in enumerate(λs)

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
        global p = Parameters(wl, φ, θ, ϵ_m, ϵ_bg)
        global IP²_factor = (p.k_1^2 - p.k_2^2) / l.V_2 / l.V .* IP²
        k_sol = scalarNewton(ks[n, mode])
        c_sol = qr(conj(getMder(k_sol,0)), Val(true)).Q[:,end]
        ksols[n, mode] = k_sol
        csols[:,n, mode] = c_sol
        println("Mode = ", mode, " | λ= ", wl)
    end
end

#Display Results as a plot ν(k) ,ksols = newton output, ks = qep output
#using Plots
#f_v = 3e5 ./ collect(wl_v)
#plot([imag.(ksols), real.(ksols)], f_v)
#plot([imag.(ks), real.(ks)], f_v)

#Write Results to txt file
# open(results_file, "w") do io
#            writedlm(io, hcat(f_v, ksols))
# end


E_x, E_y, E_z = getE_Field(600, 2, 0.5)

Eint =  abs.(E_x).^2 .+ abs.(E_y).^2 .+ abs.(E_z).^2
Eint = Eint/maximum(Eint)

using Plots
heatmap(Eint)
