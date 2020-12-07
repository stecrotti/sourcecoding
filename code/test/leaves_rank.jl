include("./../headers.jl")
include("./../SimulationNEW.jl")

q = 2
n = 1000
m = 400
bvals = [1, 10, 50, 200]
niter = 10
seed = 12

exceptions = Tuple{Int, LossyModel}[]

for (i,b) in enumerate(bvals)
    println("b = $b. $i of $(length(bvals))")
    for it = 1:niter 
        lm = LossyModel(q,n,m+b,randseed=seed+it)
        g = only_factors_graph(lm.fg)
        if length(connected_components(g)) == 1
            breduction!(lm, b, randseed=seed+it)
            rk = rank(lm)
            if rk != m
                push!(exceptions,(seed+it,lm))
            end
        end
    end
end

exceptions