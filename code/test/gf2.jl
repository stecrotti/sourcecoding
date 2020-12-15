include("./../headers.jl")
using JLD

const q = 2
const gamma = 1e-4
const n = Int(round(420*10/log2(q)))
const R = [0.7894]
const m = Int.(round.(n*(1 .- R)))
const b = Int(round(n/30))*ones(Int, length(m))
const maxiter = Int(1e3)
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

save("../../ExpectedWEF/datapoints/gf2_7.jld", "sims", sims)

unicodeplots()
plot(sims)