using PyPlot, DelimitedFiles, ColorSchemes

a=30.0
img_yrange = a
img_zrange = 0.5*sqrt(3)*a
res = 0.25

# read field data from file
# for freq in [820,844,880]
#     for mode in [1,2]
freq = 400
mode = 2
field = readdlm(string(pwd(),"\\Results\\Current_pd44_",freq,"_TM",mode,".txt"),'\n',ComplexF64)
heatmap_ys = length(collect(-img_yrange/2 : res : img_yrange/2))
heatmap_zs = length(collect(-2*img_zrange/4 : res : 2*img_zrange/4))
field = permutedims(reshape(field,(3,heatmap_ys,heatmap_zs)),[1,3,2])
# field = field[:,end:-1:1,:]

# E-Field intensity
E_I = sqrt.(dropdims(sum(abs.(field).^2,dims=1),dims=1))
Cabs = sqrt.(abs.(field[1,:,:]).^2 + abs.(field[2,:,:]).^2 + abs.(field[3,:,:]).^2)
pltnrm = maximum(E_I)

plot_data = Cabs

plt = heatmap(collect(-img_yrange/2 : res : img_yrange/2),collect(-2*img_zrange/4 : res : 2*img_zrange/4),
                        plot_data,
                        aspect_ratio=:equal,
                        proj=:polar,
                        color=:hot,
                        right_margin=5mm)
plot(plt,title = string("Current, mode: ",mode,", ",freq," THz NG=1600, pd=(4,4)"))

# integral over y-components
# plot_data = (1/a)*sum(field[2,:,:],dims=2)./sqrt.((1/a)*sum(abs.(field[2,:,:]).^2,dims=2))
# plot_data = dropdims(plot_data,dims=2)
# plot(real.(plot_data),color=:red,label="Real part",title = string("<E_y>(z), mode: ",mode,", ",2.99792458e5/parameters.lambda," THz"))
# plot!(imag.(plot_data),color=:blue,label="Imag part")
# plot!(abs.(plot_data),color=:green,label="abs()")

# Saving data, edit filenames!!
# savefig(string(pwd(),"\\Results\\Current_NG800pd44_TMk",mode,"_",freq,".png"))
# println("Saved mode ",mode," for ",freq," THz @ ",Dates.format(Dates.now(),"HH:MM"))
