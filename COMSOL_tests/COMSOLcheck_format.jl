using DelimitedFiles

for file in ["CC_BS_noKappaNG500_TM1",
             "CC_BS_R13_noKappaNG100_TM2",
             "CC_BS_R13_noKappaNG100_MGTM",
             "CC_BS_R13_noKappaNG100_MGTE",
             "CC_BS_R13_noKappaNG100_TM1",
             "CC_BS_noKappaNG500_TE",
             "CC_bandstrct_MG_R10",
             "CC_BS_R13_noKappaNG100_TE",
             "CC_BS_noKappaNG500_TM2"]

   local out = string(pwd(),"\\Results\\",file,"_reformatted.dat")

   local data = readdlm(string(pwd(),"\\Results\\",file,".dat"),'\t',skipstart=1)
   data[:,1] = convert.(ComplexF64,data[:,1])
   data[:,2] = parse.(ComplexF64,data[:,2])
   data[:,3] = parse.(ComplexF64,data[:,3])

   for i in 1:size(data)[1]
      if abs(data[i,2]-data[i,1]) > abs(data[i,3]-data[i,1])
         data[i,2]=data[i,3]
      end
   end

   open(out, "w") do io
      writedlm(io, hcat(real.(data[:,1]), real.(data[:,2]), imag.(data[:,2])))
   end
end
