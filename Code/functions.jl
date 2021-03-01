struct Parameters
    lambda::AbstractFloat
    azim::AbstractFloat
    incl::AbstractFloat
    e_m::Complex
    e_bg::AbstractFloat
    k_0::AbstractFloat
    k_x::AbstractFloat
    k_y::AbstractFloat
    k_1::Complex{Float64}
    k_2::Complex
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
    R
    B
    V
end

struct Lattice3D
    NG
    A
    V_2
    R
    B
    V
end


Lattice3D(NG,A,V_2,R) = Lattice3D(NG, A, V_2, R, 2*pi*inv(A)', abs(det(A)))

#Lattice2D(NG,A,V_2) = Lattice2D(NG, A, V_2,
#2*pi*[0 0 0; inv(A)[1,1] inv(A)[2,1] 0; inv(A)[1,2] inv(A)[2,2] 0],
#det(A))

Lattice2D(NG,A,V_2,R) = Lattice2D(NG, A, V_2, R,
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

function getHinv(Gs, k_v, k_1)::Array{Complex{Float64},5}
    #Creating 𝓗⁻¹#
    # Adding vectors
    k_v = reshape(k_v, (3,1))
    @einsum kG[i,k,n,m] := k_v[i] + Gs[i,k,n,m]
    #Creating square of elements
    @einsum kG_2[k,n,m] := kG[i,k,n,m] * kG[i,k,n,m]
    #Creating TensorProduct
    @einsum outM[i,j,k,n,m] := kG[i,k,n,m] * kG[j,k,n,m]
    #Computing inverse
    H_factor = 1 ./ (k_1^2 .- kG_2)
    outM = Matrix{ComplexF64}(I,3,3) .- outM / k_1^2
    @einsum outM[i,j,k,n,m] = H_factor[k,n,m] * outM[i,j,k,n,m]
    return outM
end

function getInitGuess(InnerP, H_inv)
    #InitialGuess
    ζ = (p.k_1^2-p.k_2^2) * l.V_2 / l.V / p.k_1^2
    @einsum Mm[i,j] :=  InnerP[k,n,m] * H_inv[i,j,k,n,m]
    Mm = I - p.k_1^2 / l.V_2^2 * ζ * Mm
    A2 = Mm - ζ * [0.0 0 0; 0 0 0; 0 0 1]
    A1 = -ζ * (p.k_x *[0.0 0 1; 0 0 0; 1 0 0] + p.k_y *[0.0 0 0; 0 0 1; 0 1 0])
    A0 = ζ * (p.k_1^2 * I - p.k_x^2 *[1.0 0 0; 0 0 0; 0 0 0] - p.k_y^2 *[0.0 0 0; 0 1 0; 0 0 0] - p.k_x * p.k_y *[0.0 1 0; 1 0 0; 0 0 0]) - (p.k_1^2 - p.k_x^2 - p.k_y^2) * Mm
    QEVP_LH = [A1 A0; -I zeros((3,3))]
    QEVP_RH = -[A2 zeros((3,3)); zeros((3,3)) I]
    return eigen(QEVP_LH,QEVP_RH)
end

function getM(λ_value, IPs, eps = 1.0e-8)::Array{Complex{Float64},2}

    k_v = [p.k_x ; p.k_y; λ_value]
    H_inv = getHinv(Gs, k_v, p.k_1)
    @einsum GH_sum[i,j,k,n,m] := IPs[k,n,m] * H_inv[i,j,k,n,m]
    GH_sum = dropdims(sum(GH_sum, dims=(3,4,5)),dims=(3,4,5))
    return I - GH_sum
end

function getMder(λ_value, IPs, eps = 1.0e-8)::Complex{Float64}

    return (det(getM(λ_value+eps, IPs)) -
            det(getM(λ_value-eps, IPs)))/(2*eps)
end

function scalarNewton(init, maxiter=1000, tol2=5e-9)
    k = init
    for nn in 1:maxiter
        phik = det(getM(k, IP²_factor))
        knew = k - phik / getMder(k, IP²_factor)
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

function getFieldValue(H, KG_slice_y, KG_slice_z, y, z)::ComplexF64

    return sum(H .* exp.(1im*KG_slice_y*y) .* exp.(1im*KG_slice_z*-z ))
end
