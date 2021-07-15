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
    lambda::AbstractFloat
    azim::AbstractFloat
    polar::AbstractFloat
    eps_1::AbstractFloat
    eps_2::Complex
    k_0::AbstractFloat
    k_x::AbstractFloat
    k_y::AbstractFloat
    k_1::Complex
    k_2::Complex
end

Parameters(lambda, azim, polar, eps_1, eps_2) = Parameters(lambda,
    azim,
    polar,
    eps_1,
    eps_2,
    2*pi/lambda,
    2*pi/lambda*cos(azim/180*pi) * sin(polar/180*pi),
    2*pi/lambda*sin(azim/180*pi) * sin(polar/180*pi),
    2*pi/lambda*sqrt(eps_1),
    2*pi/lambda*sqrt(eps_2))

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
    NG
    A
    R
    B
    V
    V_2
end

Lattice1D(NG,a,R) = Lattice(NG, a, R,2*pi/a*[0 0 0; 0 0 0; 1 0 0],a,2*R)

Lattice2D(NG,A,R) = Lattice(NG, A, R,
2*pi*[0 0 0; inv(A)[1,1] inv(A)[2,1] 0; inv(A)[1,2] inv(A)[2,2] 0],abs(det(A)),pi*R^2)

Lattice3D(NG,A,R) = Lattice(NG, A, R, 2*pi*inv(A)', abs(det(A)),3/4*pi*R^3)
