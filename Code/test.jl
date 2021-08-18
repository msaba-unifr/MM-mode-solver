using Distributed,BenchmarkTools
if nprocs() < 2
    addprocs(7)
end
@everywhere using LinearAlgebra

@everywhere function my_func(x)
    A = rand(100,100)
    eigen(A)
end

@btime my_func.(1:100)
@btime pmap(my_func,1:100,batch_size=1)
@btime pmap(my_func,1:100,batch_size=10)
@btime pmap(my_func,1:100   ,batch_size=100)
