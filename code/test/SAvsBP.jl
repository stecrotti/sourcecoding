# using Plots
# unicodeplots()
using Printf, Plots

include("./../headers.jl")
include("./../SimulationNEW.jl")

q = 2
n = 1000
mvals = Int.(round.(n*(0.2:0.1:0.9)))
b = 10
niter = 20
randseed = 123

sa = SA(mc_move=MetropSmallJumps(), nsamples=1000, betas=[Inf 100.0])
ms = MS(maxiter=200, gamma=5e-3)

sim_sa = Vector{Simulation{SA}}(undef, length(mvals))
sim_ms = Vector{Simulation{MS}}(undef, length(mvals))

for (j,m) in enumerate(mvals)
    sim_ms[j] = Simulation(q,n,m,ms,b=b, niter=niter, verbose=true, 
        randseed=randseed)
    sim_sa[j] = Simulation(q,n,m,sa,b=b, niter=niter, verbose=true, 
        randseed=randseed)
    println("\n### Finished m=", m, " ###")
    println("Distortion SA: ", @sprintf("%.2f", distortion(sim_sa[j])))
    println("Distortion MS: ", @sprintf("%.2f", distortion(sim_ms[j])))
    println()
end

pl = plot(sim_sa, label="SA")
plot!(pl, sim_ms, label="MS")