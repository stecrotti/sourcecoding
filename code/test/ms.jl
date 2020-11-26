using Printf, Plots
unicodeplots()

include("./../headers.jl")
include("./../SimulationNEW.jl")

const q = 2
const n = 1200
const mvals = reverse(Int.(round.(n*(0.2:0.35:0.9))))
const b = Int(round(n/30))
const niter = 30
const randseed = 1234

sims = Vector{Simulation{MS}}(undef, length(mvals))
ms = MS(maxiter=500, gamma=1e-2)

for (j,m) in enumerate(mvals)
    println("##### R = ", 1-n/m, " #####\n")
    sims[j] = Simulation(q, n, m, ms, b=b, randseed=randseed+200*j, 
        verbose=true)
end

plot(sims)