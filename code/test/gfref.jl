using Printf, Plots
unicodeplots()

include("./../headers.jl")
include("./../SimulationNEW.jl")

const q = 2
nvals = Int.(floor.(10 .^(LinRange(1,3.5,7))))
R = 0.3
mvals = ceil.(Int, nvals.*(1-R))
niter = 50
randseed = 1234
# nleaves = [zeros(niter) for n in nvals]
variable_degrees = [[Dict{Int64,Float64}() for it in 1:niter] for n in 1:length(nvals)]
nleaves = [zeros(Int, niter) for n in 1:length(nvals)]

for (i,n) in enumerate(nvals)
    println("n=$n. $i of $(length(nvals))")
    for it in 1:niter
        lm = LossyModel(q, n, mvals[i], randseed=randseed+niter*i+it)
        gfref!(lm)
        nleaves[i][it] = nvarleaves(lm.fg)
        variable_degrees[i][it] = vardegrees_distr(lm.fg)
    end
end

vd_vec = [Float64[] for n in 1:length(nvals)]

avg_leaves = [mean(nleaves[j]) for j in eachindex(nvals)]
sd_leaves = [std(nleaves[j])/sqrt(niter) for j in eachindex(nvals)]
plt = Plots.plot(nvals, avg_leaves, markershape=:circle, label="", ribbon=sd_leaves)
xlabel!("n")
ylabel!("Number of leaves")
title!("Number of leaves vs n")

for i in 1:length(nvals)
    m = maximum([maximum(collect(keys(variable_degrees[i][j]))) 
        for j in eachindex(variable_degrees[i])])
    vd_vec[i] = [zeros(m) for _ in 1:niter]
    for j in eachindex(variable_degrees[i])
        for k in keys(variable_degrees[i][j])
            vd_vec[i][j][k] = variable_degrees[i][j][k]
        end
    end
end