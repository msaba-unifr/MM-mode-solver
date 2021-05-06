using PlyIO

function write_disk(nverts, r)
    ply = Ply()
    push!(ply, PlyComment("An example disk ply file"))

    r = 10
    dθ = 2*pi/nverts

    vertex_x = zeros(nverts)
    vertex_y = [r*cos(n*dθ) for n in 0:nverts-1]
    #vertex_y = append!([0.0], vertex_y)
    vertex_z = [r*sin(n*dθ) for n in 0:nverts-1]
    #vertex_z = append!([0.0], vertex_z)

    # Random vertices with position
    vertex = PlyElement("vertex",
                        ArrayProperty("x", vertex_x),
                        ArrayProperty("y", vertex_y),
                        ArrayProperty("z", vertex_z))
    push!(ply, vertex)

    # Some triangular faces.
    # The UInt8 is the type used for serializing the number of list elements (equal
    # to 3 for a triangular mesh); the Int32 is the type used to serialize indices
    # into the vertex array.
    vertex_index = ListProperty("vertex_indices", UInt8, Int32)

    push!(vertex_index, collect(0:nverts-1))

    push!(ply, PlyElement("face", vertex_index))



    # For the sake of the example, ascii format is used, the default binary mode is faster.
    save_ply(ply, "example2.ply", ascii=true)

    #using Plots

    #scatter(vertex_x, vertex_y)
end
