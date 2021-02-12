using LinearAlgebra
using Einsum
using SpecialFunctions
using NonlinearEigenproblems
using DelimitedFiles
using Interpolations
include("functions.jl")

#test for schumi commit
#Parameters set by the user
wl = 750
φ = 90 #for k_x
θ = 10 #for k_y
#ϵ_m = (0.06 + 4.152im)^2
NG = 10
ϵ_bg = 1 + 0im
epsilon_m_file = "MaterialModels/Ag_JC_nk.txt"

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


#Code starts here
l = Lattice2D(NG,A,V_2)
#Creating G_space
Gs = getGspace()
IP_1 = dropdims(InnerProd.(sqrt.(sum(Gs.*Gs,dims=1))*R,exclude_DC=true),dims=1).^2
IP_2 = dropdims(InnerProd.(sqrt.(sum(Gs.*Gs,dims=1))*R),dims=1).^2

#Preallocating for result arrays
nmode = 2
wl_v = 750.0 : -Δλ : 700
ks = zeros(ComplexF64, (size(wl_v,1),nmode))
cs = zeros(ComplexF64, (3,size(wl_v,1),nmode))
ksols = zeros(ComplexF64, (size(wl_v,1),nmode))
csols = zeros(ComplexF64, (3,size(wl_v,1),nmode))

#Interpolating (n,k) data from txt file
eps_data = readdlm(epsilon_m_file, '\t', Float64, '\n')
itp1 = LinearInterpolation(eps_data[:,1], eps_data[:,2])
itp2 = LinearInterpolation(eps_data[:,1], eps_data[:,3])
eps_ms = (itp1.(wl_v ./ 1000) + itp2.(wl_v ./ 1000) * 1im).^2

#Loop for calculating all initial guesses in wl_v
for (n, wl) in enumerate(wl_v)
    ϵ_m = eps_ms[n]
    global p = Parameters(wl, φ, θ, ϵ_m, ϵ_bg)

    #Creating 𝓗⁻¹ for initial guess
    k_vec = zeros(ComplexF64, (3,1))
    𝓗inv = getHinv(Gs,k_vec)
    𝓗Ginv = getHinv(Gs,[0;0;0])
    #InitialGuess
    eigs_init = getInitGuess(IP_1,𝓗Ginv)
    global λs,vs = eigs_init.values, eigs_init.vectors
    println(eigs_init.values)

    nλ = 1
    tol = 1e-3
    for (i, λ_val) in enumerate(λs)

        #Eigenvalue filtering, only works for Φ, Θ = 0 yet
        if real(λ_val) <= 0
            println("Skipped negative lambda")
            continue
        elseif abs(1- (λ_val^2+p.k_x^2+p.k_y^2)/(p.e_bg*p.k_0^2)) < 1e-8
            println("Skipped spurious lambda")
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
f_v = 3e5 ./ collect(wl_v)
plot([imag.(ksols), real.(ksols)], f_v)
#plot([imag.(ks), real.(ks)], f_v)

#Write Results to txt file
open(results_file, "w") do io
           writedlm(io, hcat(f_v, ksols))
end
