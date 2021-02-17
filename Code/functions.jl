struct Parameters
    lambda
    azim
    incl
    e_m
    e_bg
    k_0
    k_x
    k_y
    k_1
    k_2
end

Parameters(lambda, azim, incl, e_m, e_bg) = Parameters(lambda,azim,incl,e_m,e_bg,
    2*pi/lambda,
    2*pi/lambda*cos(azim/180*pi) * sin(incl/180*pi),
    2*pi/lambda*sin(azim/180*pi) * sin(incl/180*pi),
    2*pi/lambda*sqrt(e_bg),
    2*pi/lambda*sqrt(e_m))

struct Lattice2D
    NG
    A
    V_2
    B
    V
end

struct Lattice3D
    NG
    A
    V_2
    B
    V
end


Lattice3D(NG,A,V_2) = Lattice3D(NG, A, V_2, 2*pi*inv(A)', abs(det(A)))

#Lattice2D(NG,A,V_2) = Lattice2D(NG, A, V_2,
#2*pi*[0 0 0; inv(A)[1,1] inv(A)[2,1] 0; inv(A)[1,2] inv(A)[2,2] 0],
#det(A))

Lattice2D(NG,A,V_2) = Lattice2D(NG, A, V_2,
2*pi*[0 0 0; inv(A)[1,1] inv(A)[2,1] 0; inv(A)[1,2] inv(A)[2,2] 0],
abs(det(A)))

function InnerProd(x;exclude_DC=false)
    if x == 0
        if exclude_DC
            return 0
        else
            return l.V_2  # 2 * (1/2)
        end
    end
    #return 2 / x * besselj(1,x)
    return 2 * l.V_2 / x * besselj(1,x)
end

# In the 2D case:
function getGspace()
    #Creating G_space
    space_ax = -l.NG:1.0:l.NG
    Gs = [l.B*[h,k,n] for h in space_ax, k in space_ax, n in 0:0]
    Gs = [Gs[h,k,n][i] for i in 1:3, h in 1:2*l.NG+1, k in 1:2*l.NG+1, n in 1:1]
    return Gs
end

function getHinv(Gs,k_v)
    #Creating 𝓗⁻¹#
    # Adding vectors
    k_v = reshape(k_v, (3,1))
    @einsum kG[i,k,n,m] := k_v[i] + Gs[i,k,n,m]
    #Creating square of elements
    @einsum kG_2[k,n,m] := kG[i,k,n,m] * kG[i,k,n,m]
    #Creating TensorProduct
    @einsum kG_TP[i,j,k,n,m] := kG[i,k,n,m] * kG[j,k,n,m]
    #Computing inverse
    H_factor = 1 ./ (p.k_1^2 .- kG_2)
    id_TP = Matrix{ComplexF64}(I,size(kG_TP,1),size(kG_TP,2)) .- kG_TP / p.k_1^2
    @einsum H_inv[i,j,k,n,m] := H_factor[k,n,m] * id_TP[i,j,k,n,m]
    return H_inv
end

function getInitGuess(InnerP,H_inv)
    #InitialGuess
    ζ = (p.k_1^2-p.k_2^2) * l.V_2 / l.V / p.k_1^2
    @einsum Mm[i,j] :=  InnerP[k,n,m] * H_inv[i,j,k,n,m]
    Mm = I - p.k_1^2 / V_2^2 * ζ * Mm
    A2 = Mm - ζ * [0.0 0 0; 0 0 0; 0 0 1]
    A1 = -ζ * (p.k_x *[0.0 0 1; 0 0 0; 1 0 0] + p.k_y *[0.0 0 0; 0 0 1; 0 1 0])
    A0 = ζ * (p.k_1^2 * I - p.k_x^2 *[1.0 0 0; 0 0 0; 0 0 0] - p.k_y^2 *[0.0 0 0; 0 1 0; 0 0 0] - p.k_x * p.k_y *[0.0 1 0; 1 0 0; 0 0 0]) - (p.k_1^2 - p.k_x^2 - p.k_y^2) * Mm
    QEVP_LH = [A1 A0; -I zeros((3,3))]
    QEVP_RH = -[A2 zeros((3,3)); zeros((3,3)) I]
    return eigen(QEVP_LH,QEVP_RH)
end

function getMder(λ_value, der, eps = 1.0e-8)

    k_v = [p.k_x ; p.k_y; λ_value]

    if der == 0
        H_inv = getHinv(Gs, k_v)
        @einsum GH_sum[i,j,k,n,m] := IP²_factor[k,n,m] * H_inv[i,j,k,n,m]
        GH_sum = dropdims(sum(GH_sum, dims=(3,4,5)),dims=(3,4,5))
        return I - GH_sum
    elseif der == 1 #Recursive implementation of numeric derivative
        return (det(getMder(λ_value+eps,der-1)) -
            det(getMder(λ_value-eps,der-1)))/(2*eps)
    end
end

function scalarNewton(init, maxiter=1000, tol2=5e-9)

    k = init
    for nn in 1:maxiter
        phik = det(getMder(k,0))
        knew = k - phik / getMder(k,1)
        delk = abs(1 - knew / k)
        if delk < tol2
            global knew = knew
            break
        end
        k = knew
        nn == maxiter ? throw("No conv in Newton") : nothing
    end
    return knew
end

function getE_Field(wl_input, nmode, res)

    #find closest computed wl index with respect to input wl
    n = findmin(abs.(wl_v .- wl_input))[2]
    println("Closeset wavelength computed: ", wl_v[n])
    #update dependencies with respect to desired wl
    ϵ_m = eps_ms[n]
    global p = Parameters(wl_v[n], φ, θ, ϵ_m, ϵ_bg)
    #Extract eigen-values/vectors from Newton step
    k_sol = ksols[n, nmode]
    c_sol = csols[:, n, nmode]
    #image range dependent on lattice parameter
    img_yrange = 2*a #nm
    img_zrange = sqrt(3)*a #nm
    ys = -img_yrange/2 : res : img_yrange/2
    zs = -img_zrange/2 : res : img_zrange/2
    #Precomputing variables
    kpGs = [p.k_x, p.k_y, k_sol] .+ Gs
    HikG = getHinv(Gs, [p.k_x, p.k_y, k_sol])
    absGs = dropdims(sqrt.(sum(Gs.^2,dims=1)),dims=1)
    #Calculation according to manuscript
    @einsum H_c[i,k,n,m] := IP[k,n,m] * HikG[i,j,k,n,m] * c_sol[j]
    H_c = H_c ./ l.V
    #Field components for every z-y position in image range
    E_x = [sum(H_c[1,:,:,:] .*
        exp.(1im*kpGs[2,:,:,:]*y) .* exp.(1im*kpGs[3,:,:,:]*-z )) for z in zs, y in ys]
    E_y = [sum(H_c[2,:,:,:] .*
        exp.(1im*kpGs[2,:,:,:]*y) .* exp.(1im*kpGs[3,:,:,:]*-z )) for z in zs, y in ys]
    E_z = [sum(H_c[3,:,:,:] .*
        exp.(1im*kpGs[2,:,:,:]*y) .* exp.(1im*kpGs[3,:,:,:]*-z )) for z in zs, y in ys]

    return E_x, E_y, E_z
end
