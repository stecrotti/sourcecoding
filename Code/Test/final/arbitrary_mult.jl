include("../../headers.jl")

const q = 16
const gamma = 5e-3
const n = Int(round(420*2/log2(q)))
const R = collect(0.1:0.1:0.9)
const m = Int.(round.(n*(1 .- R)))
const b = Int(round(n/30))*ones(Int, length(m))
const maxiter = Int(5e2)
const navg = 200
const randseed = 43
const Tmax = 4

sims_gfq = Vector{Simulation}(undef, length(m))
sims_arb = Vector{Simulation}(undef, length(m))

for j in 1:length(m)
    println("---------- Simulation $j of ", length(m)," | R = ",R[j]," -----------")
    sims_gfq[j] = Simulation(MS(), q, n, m[j],
        navg=navg, convergence=:parity, maxiter=maxiter, gamma=gamma, Tmax=Tmax,
        b=b[j], samegraph=false, samevector=false, randseed=randseed+navg*Tmax*j,
        verbose=true, arbitrary_mult = false)
    print(sims_gfq[j])
end

for j in 1:length(m)
    println("---------- Simulation $j of ", length(m)," | R = ",R[j]," -----------")
    sims_arb[j] = Simulation(MS(), q, n, m[j],
        navg=navg, convergence=:parity, maxiter=maxiter, gamma=gamma, Tmax=Tmax,
        b=b[j], samegraph=false, samevector=false, randseed=randseed+navg*Tmax*j,
        verbose=true, arbitrary_mult = true)
    print(sims_arb[j])
end

date = Dates.format(now(), "yyyymmdd_HHMM")
save("arbitrary-"*date*".jld", "sims_gfq", sims_gfq, "sims_arb", sims_arb)
