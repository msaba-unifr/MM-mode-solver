"""
    Stores generic parameters not changed during runtime

Input parameters (lengths in nm and angles in degrees):
    lambda = wavelength
    azim   = azimuthal angle of wave in physics convention
    polar  = polar  angle of wave in physics convention
    eps_1  = real permittivity in the background medium
    eps_2  = complex permittivity in the cylinder
Derived parameters (wave numbers in 1/nm):
    k_0    = vacuum wavenumber
    k_x    = x component of wave vector
    k_y    = y component of wave vector (should store those two in vector?)
    k_1    = wave number in background domain
    k_2    = wave number in cylinder domain
"""
struct Parameters
    lambda::Float64
    azim::Float64
    polar::Float64
    eps_1::Float64
    eps_2::ComplexF64
    polydegs::Tuple{Int64, Int64}
    material::String
    k_0::Float64
    k_x::Float64
    k_y::Float64
    k_1::ComplexF64
    k_2::ComplexF64
    function Parameters(lambda, azim, polar, eps_1, eps_2, polydegs, mat_file)
        k_0 = 2*pi/lambda
        k_x = k_0 * cos(azim/180*pi) * sin(polar/180*pi)
        k_y = k_0 * sin(azim/180*pi) * sin(polar/180*pi)
        k_1 = k_0 * sqrt(eps_1)
        k_2 = k_0 * sqrt(eps_2)
        material = mat_file[1:2]
        new(lambda,azim,polar,eps_1,eps_2,polydegs,material,k_0,k_x,k_y,k_1,k_2)
    end
end

"""
    Stores geometrical and lattice parameters.

Input parameters (length in nm, volume in nm^d, d=dimension of lattice):
    NG  = cut-off for recirocal lattice vectors. G_{hkl}=B.(h,k,l), h,k,l in -NG:NG
    A   = dxd real space lattice matrix with primitive lattice vectors as columns
    R   = radius of d-sphere defining domain 2
Deriver parameters (reciprocal lattice vectors in 1/nm, length in nm, volume in nm^d):
    B   = dxd reciprocal lattice matrix with primitive lattice vectors as columns such that G_{hkl}=B.(h,k,l)
    V   = volume of primitive unit cell
    V_2 = volume of domain 2
"""
struct Lattice
    NG::Int
    Gs::Array{Array{Float64,1},3}
    A::Array{Float64,2}
    R::Float64
    B::Array{Float64,2}
    V::Float64
    V_2::Float64
    function Lattice(NG::Int,A::Array{Float64,2},R::Float64)
        B = 2*pi*[0 0 0; inv(A)[1,1] inv(A)[2,1] 0; inv(A)[1,2] inv(A)[2,2] 0]
        Gs = [B*[h,k,n] for h in -NG:1:NG, k in -NG:1:NG, n in 0:0]
        V = abs(det(A))
        V_2 = pi*R^2
        new(NG,Gs,A,R,B,V,V_2)
    end
end
