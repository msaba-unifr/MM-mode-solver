function gabard_integral_2D(Hnr, k_vec, ver_x, ver_y, ver_z, faces)::ComplexF64
    #ply = load_ply(string(pwd(), "\\example1.ply"))

    k_gab = 1im*k_vec

    N = size(Hnr, 1)-2
    R = size(Hnr, 2)-2
    max_d = maximum([N, R])
    binom = zeros((max_d+1, max_d+1))
    for i in 1:max_d+1, j in 1:max_d+1
        binom[i,j] = binomial(i,j)
    end
    kv_norm = norm(k_gab)

    integral = 0.0 + 0im
    for face in faces
        points = [[ver_x[idx+1], ver_y[idx+1], ver_z[idx+1]] for idx in face]
        integral += CalcFace(points, Hnr, N, R, max_d, k_gab, kv_norm, binom)
    end

    return integral
end

function CalcFace(points, Hmnr, N, R, max_d, k_v, kv_norm, binom)::ComplexF64

    n_v = cross(points[2] - points[1], points[3] - points[1])
    n_v = n_v / norm(n_v)

    a_v = cross(n_v, conj(k_v))/(kv_norm^2)
    J_sum = 0.0 + 0im

    for i in 1:length(points)

        x0 = points[i]
        x1 = points[mod(i,length(points))+1] - x0
        ψ = dot(k_v, x1)
        if abs(ψ) > 1e-8 #add as tolerance
            Qn = zeros(ComplexF64, N+R+1)
            Qn[1] = (exp(ψ)-1) / ψ
            for i in 2:N+R+1

                Qn[i] = (exp(ψ) - (i-1)*Qn[i-1])/ψ
            end
        else
            Qn = 1 ./ (collect(0:(N+R)) .+ 1)
        end

        y0p = zeros(max_d+1)
        y0p[1] = 1.0
        for i in 2:max_d+1

            y0p[i] = y0p[i-1]*x0[2]
        end

        z0p = zeros(max_d+1)
        z0p[1] = 1.0
        for i in 2:max_d+1

            z0p[i] = z0p[i-1]*x0[3]
        end

        y1p = zeros(max_d+1)
        y1p[1] = 1.0
        for i in 2:max_d+1

            y1p[i] = y1p[i-1]*x1[2]
        end

        z1p = zeros(max_d+1)
        z1p[1] = 1.0
        for i in 2:max_d+1

            z1p[i] = z1p[i-1]*x1[3]
        end


        Gnr = zeros(Float64, (max_d+2, max_d+2))
        for n in 0:N, r in 0:R
            #idx m = y coordinate, idx n = z coordinate
            Gnr[n+1,r+1] = (Hmnr[n+1,r+1] -
                (n+1)*(a_v[3]*n_v[1] - a_v[1]*n_v[3])*Gnr[n+2,r+1] -
                (r+1)*(a_v[1]*n_v[2] - a_v[2]*n_v[1])*Gnr[n+1,r+2]) *
                1/(dot(a_v, cross(n_v, k_v)))
        end

        J_sum  += dot(x1, a_v) * exp(dot(k_v, x0)) *
            CalcSegSum(N, R, Gnr, binom, y0p, z0p, y1p, z1p, Qn, max_d)
    end
    return J_sum
end

function CalcSegSum(N, R, Gmnr, binM, y0M, z0M, y1M, z1M, QnM, max_d)::ComplexF64

    out_sum = 0.0 + 0im
    for n in 0:N, r in 0:R

        if Gmnr[n+1, r+1] != 0
            in_sum = 0
            for v in 0:n, w in 0:r

                in_sum += binM[n+1,v+1] * binM[r+1,w+1] * y0M[n-v+1] *
                    z0M[r-w+1] * y1M[v+1] * z1M[w+1] * QnM[v+w+1]

            end
            out_sum += Gmnr[n+1,r+1]*in_sum
            #println("G",n-1, r-1,"= ", Gmnr[max_d-n+3,max_d-r+3], "  |   insum= ", in_sum)
        end
    end
    return out_sum
end
