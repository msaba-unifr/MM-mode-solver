using Plots, Plots.PlotMeasures, DelimitedFiles, ColorSchemes

freqs = collect(850.0:0.1:852.0)
rads = collect(9:-0.01:8.5)

tm1_reals = zeros(Float64,(21,51))
tm1_imags = zeros(Float64,(21,51))
tm2_reals = zeros(Float64,(21,51))
tm2_imags = zeros(Float64,(21,51))

tm1 = zeros(ComplexF64,(21,51))
tm2 = zeros(ComplexF64,(21,51))

for (nf,freq) in enumerate(freqs)
    tm1_reals[nf,:] = readdlm(string(pwd(),"\\Results\\Ag_Rad9to85_TM1_",freq,".dat"),'\t',Float64,skipstart=3)[:,2]
    tm1_imags[nf,:] = readdlm(string(pwd(),"\\Results\\Ag_Rad9to85_TM1_",freq,".dat"),'\t',Float64,skipstart=3)[:,3]
    tm2_reals[nf,:] = readdlm(string(pwd(),"\\Results\\Ag_Rad9to85_TM2_",freq,".dat"),'\t',Float64,skipstart=3)[:,2]
    tm2_imags[nf,:] = readdlm(string(pwd(),"\\Results\\Ag_Rad9to85_TM2_",freq,".dat"),'\t',Float64,skipstart=3)[:,3]

    tm1[nf,:] = tm1_reals[nf,:]+tm1_imags[nf,:]*im
    tm2[nf,:] = tm2_reals[nf,:]+tm2_imags[nf,:]*im
end

tm1_reals = tm1_reals[:,end:-1:1]
tm1_imags = tm1_imags[:,end:-1:1]
tm2_reals = tm2_reals[:,end:-1:1]
tm2_imags = tm2_imags[:,end:-1:1]

# plot_data_reals = tm1_reals_swap-tm2_reals_swap
# plot_data_imags = tm1_imags_swap-tm2_imags_swap
#
# p1 = contour(collect(8.5:0.01:9),collect(850:0.1:852),plot_data_reals,levels=[0],fill=false,c=:red)
# contour!(collect(8.5:0.01:9),collect(850:0.1:852),plot_data_imags,levels=[0],fill=false,c=:blue)

camera_angle = (30,30)

# plot(collect(8.5:0.01:9),collect(850:0.1:852),tm1_reals,st=:surface,fillalpha=0.5,camera=camera_angle)
# plot!(collect(8.5:0.01:9),collect(850:0.1:852),tm2_reals,st=:surface,fillalpha=0.5)
# savefig(string(pwd(),"\\Results\\Surfaces_reals_",camera_angle,".png"))

plot(collect(8.5:0.01:9),collect(850:0.1:852),tm1_imags,st=:surface,fillalpha=0.5,camera=camera_angle)
plot!(collect(8.5:0.01:9),collect(850:0.1:852),tm2_imags,st=:surface,fillalpha=0.5)
# savefig(string(pwd(),"\\Results\\Surfaces_imags_",camera_angle,".png"))
