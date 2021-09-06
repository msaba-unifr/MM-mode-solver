using NonlinearEigenproblems

function Mder(λ, der)
#This funcion computes the 0th / 1st order derivative of our problem T(λ)
    deg = polydegs
    if der == 0
        out = getpolyxM(deg, λ, l, p)
    end
    if der == 1
        out = getpolyxMder_NEP_PACK(deg, λ, l, p)
    end
    return out
end

function getpolyxMder_NEP_PACK(deg,λ_value, l::Lattice, p::Parameters, eps = 1.0e-12)
#Here the derivative ist output as a matrix (vs as a scalar in the manual newton)
    return (getpolyxM(deg,λ_value+eps, l, p) .- getpolyxM(deg,λ_value-eps, l, p))./(2*eps)
end

function NEP_PACK_newton(deg, kinit, l::Lattice, p::Parameters, maxiter=1000, tol2=5e-9)
#Creates the NEP object and solves it with NEP-PACK Newton
    nep = Mder_NEP(3*(deg[1]+1)*(deg[2]+1), Mder, maxder=1)
    λ, v = augnewton(nep, tol=tol2, maxit=maxiter, λ=kinit, logger=2)
    return λ
end

function getpolyxMode_NEP_PACK(deg, l::Lattice, p::Parameters; manual_ks=[0im,0im])
#Same function as getpolyxMode except for Newton call
    ks = zeros(ComplexF64,(2))
    dim = 3*((deg[1]+1)*(deg[2]+1))
    if norm(manual_ks) != 0
        ks = manual_ks
    end

    #Solve NLEVP for each non filtered mode
    ksols = zeros(ComplexF64,(2))
    csols = zeros(ComplexF64,(dim,2))
    for mode = 1:size(manual_ks,1)
        k_sol = NEP_PACK_newton(deg,ks[mode], l, p)
        c_sol = qr(transpose(conj(getpolyxM(deg,k_sol, l, p))), Val(true)).Q[:,end]
        ksols[mode] = k_sol
        csols[:, mode] = c_sol
    end
    return ksols, csols
end
