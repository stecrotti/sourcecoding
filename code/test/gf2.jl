include("./../headers.jl")
# include("./../SimulationNEW.jl")
using JLD

const q = 2
const gamma = 1e-4
const n = Int(420*10/log2(q))
const R = [0.8]
const m = Int.(round.(n*(1 .- R)))
const b = Int(round(n/20))*ones(Int, length(m))
const maxiter = Int(5e2)
const niter = 20
const randseed = 1234
const Tmax = 8

ms = MS(maxiter=maxiter, gamma=gamma, Tmax=Tmax)
sims = Vector{Simulation{MS}}(undef, length(m))

for j in 1:length(m)
    println("\n---------- Simulation $j of ", length(m)," | R = ",R[j]," -----------")
    sim = Simulation(q, n, m[j], ms, niter=niter, b=b[j], randseed=randseed, 
        showprogress=false)
    print(sim)
    sims[j] = sim
end

save("gf2_6.jld", "sims", sims)

unicodeplots()
plot(sims)