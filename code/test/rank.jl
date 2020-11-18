include("./../headers.jl")

q = 2
n = 200
mvals = [50 100 150 180]
navg = 100
seed = 12345

exceptions = LossyModel[]

for m in mvals
    for i = 1:navg  
        lm = LossyModel(q,n,m,randseed=seed+i)
        if rank(lm) != m-1
            g = only_factors_graph(lm.fg)
            if length(connected_components(g))!=m-rank(lm)
                push!(exceptions,lm)
            end
        end
    end
    println("m = $m finished")
end
