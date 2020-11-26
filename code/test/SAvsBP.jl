using Printf, Plots
unicodeplots()

include("./../headers.jl")
include("./../SimulationNEW.jl")

const q = 2
const n = 2000
const mvals = reverse(Int.(round.(n*(0.25:0.15:0.7))))
const b = Int(round(n/30))
const niter = 30
const randseed = 1234

sa = SA(mc_move=MetropSmallJumps(), nsamples=100, 
    betas=[Inf 0.1; Inf 1.0; Inf 10.0;])
ms = MS(maxiter=200, gamma=5e-3)

sim_sa = Vector{Simulation{SA}}(undef, length(mvals))
sim_ms = Vector{Simulation{MS}}(undef, length(mvals))

for (j,m) in enumerate(mvals)
    println("\n####  m=", m, " ####\n")
    sim_sa[j] = Simulation(q,n,m,sa,b=b, niter=niter, verbose=true, 
        randseed=randseed)
    sim_ms[j] = Simulation(q,n,m,ms,b=b, niter=niter, verbose=true, 
        randseed=randseed)
    println("\n### Finished m=", m, " ###")
    println("Distortion SA: ", @sprintf("%.2f", distortion(sim_sa[j])))
    println("Distortion MS: ", @sprintf("%.2f", distortion(sim_ms[j])))
    println()
end

pl = plot(sim_sa, label="SA")
plot!(pl, sim_ms, label="MS")