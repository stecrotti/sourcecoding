include("../../headers.jl")

const q = 2
const gamma = 5e-3
const n = Int(420*10/log2(q))
const R = collect(0.1:0.1:0.9)
const m = Int.(round.(n*(1 .- R)))
const b = Int(round(n/30))*ones(Int, length(m))
const maxiter = Int(7e2)
const navg = 200
const randseed = 100
const Tmax = 6

sims = Vector{Simulation}(undef, length(m))

for j in 1:length(m)
    println("---------- Simulation $j of ", length(m)," | R = ",R[j]," -----------")
    sim = Simulation(MS(), q, n, m[j],
        navg=navg, convergence=:parity, maxiter=maxiter, gamma=gamma, Tmax=Tmax,
        b=b[j], samegraph=false, samevector=false, randseed=(randseed+navg*Tmax*j)*q,
        verbose=true)
    print(sim)
    sims[j] = sim
end

date = Dates.format(now(), "yyyymmdd_HHMM")
save("gf$q-"*date*".jld", "sims", sims)

print(sims)
