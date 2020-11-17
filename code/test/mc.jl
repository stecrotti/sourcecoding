using Plots
unicodeplots()

include("./../headers.jl")

const q = 2
const n = 100
const m = 20
const b = 1
const niter = 20
const randseed = 1234

results_nored = Vector{SAResults}(undef, niter)
results_red = Vector{SAResults}(undef, niter)
ranks_nored = zeros(Int, niter)
ranks_red = zeros(Int, niter)

for it in 1:niter
    lm_nored = LossyModel(q,n,m, randseed=randseed+it)
    lm_red = deepcopy(lm_nored)
    algo_nored = SA(lm_nored)
    results_nored[it] = solve!(lm_nored, algo_nored, randseed=randseed+it)
    ranks_nored[it] = n - size(algo_nored.mc_move.basis,2)
    
    breduction!(lm_red, b)
    algo_red = SA(lm_red)
    results_red[it] = solve!(lm_red, algo_red, randseed=randseed+it)
    ranks_red[it] = n - size(algo_red.mc_move.basis,2)
end

dist_nored = [r.distortion for r in results_nored]
dist_red = [r.distortion for r in results_red]

plot(dist_nored, label="No reduction")
plot!(dist_red, label="Reduction")