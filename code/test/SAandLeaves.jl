using Printf, Plots

include("./../headers.jl")
include("./../SimulationNEW.jl")

const q = 2
const n = 1000
const m = 800
const bvals = [1, 3, 5, 10, 20, 50]
const niter = 20
const randseed = 1234

sims = Vector{Simulation{SA}}(undef, length(bvals))
sa = SA(mc_move=MetropSmallJumps(), nsamples=100, 
    betas=[Inf 0.1; Inf 1.0; Inf 10.0;])

for (i,b) in enumerate(bvals)
    println("### b=$b. $i of $(length(bvals))###")
    sims[i] = Simulation(q,n,m,sa,b=b, niter=niter, verbose=true, 
    randseed=randseed)
end

unicodeplots()
pl = plot(sims, label="SA")

