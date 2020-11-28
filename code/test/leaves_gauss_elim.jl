using Printf, Plots
unicodeplots()

include("./../headers.jl")
include("./../SimulationNEW.jl")

const q = 2
const n = 500
const mvals = reverse(Int.(round.(n*(0.2:0.15:0.95))))
const b = Int(round(n/50))
const niter = 20
const randseed = 1234

sims1 = Vector{Simulation{MS}}(undef, length(mvals))
sims2 = Vector{Simulation{MS}}(undef, length(mvals))
ms = MS(maxiter=500, gamma=1e-2)

for (j,m) in enumerate(mvals)
    println("##### R = ", round(1-m/n, digits=2), ". ",
    "$j of $(length(mvals)) #####\n")
    
    sims2[j] = Simulation(q, n, m, ms, b=0, gauss_elim=true,
        randseed=randseed+200*j, verbose=true)
    sims1[j] = Simulation(q, n, m, ms, b=b, randseed=randseed+200*j, 
        verbose=true)
end

plot(sims1)
plot(sims2)