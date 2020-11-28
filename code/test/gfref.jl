using Printf, Plots
unicodeplots()

include("./../headers.jl")
include("./../SimulationNEW.jl")

const q = 2
nvals = Int.(floor.(10 .^(LinRange(1,3.1,7))))
R = 0.3
mvals = ceil.(Int, nvals.*(1-R))
niter = 50
randseed = 1234
nleaves = [zeros(niter) for n in nvals]

for (i,n) in enumerate(nvals)
    println("n=$n. $i of $(length(nvals))")
    for it in 1:niter
        lm = LossyModel(q, n, mvals[i], randseed=randseed+niter*i+it)
        gfref!(lm)
        nleaves[i][it] = nvarleaves(lm.fg)
    end
end

avg_leaves = [mean(nleaves[j]) for j in eachindex(nvals)]
sd_leaves = [std(nleaves[j])/sqrt(niter) for j in eachindex(nvals)]
Plots.plot(nvals, avg_leaves, markershape=:circle, label="", ribbon=sd_leaves)
xlabel!("n")
ylabel!("Number of leaves")
title!("Number of leaves vs n")