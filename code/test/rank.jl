include("./../headers.jl")

q = 2
n = 500
mvals = [100 200 300 400]
navg = 100
seed = 12345

exceptions = LossyModel[]

for m in mvals
    for i = 1:navg  
        lm = LossyModel(q,n,m,randseed=seed+i)
        rk = rank(lm)
        if rk != m-1
            g = only_factors_graph(lm.fg)
            if length(connected_components(g))!=m-rk
                push!(exceptions,lm)
            end
            breduction!(lm)
            if rank(lm) != rk
                push!(exceptions,lm)
            end
        end
    end
    println("m = $m finished")
end

exceptions