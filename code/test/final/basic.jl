include("../../headers.jl")

const q = 2
const gamma = 0
const n = Int(420*10/log2(q))
const R = collect(0.1:0.1:0.9)
const m = Int.(round.(n*(1 .- R)))
const b = Int(round(n/50))*ones(Int, length(m))
const maxiter = Int(5e2)
const navg = 100
const randseed = 12345
const Tmax = 6

sims = Vector{Simulation}(undef, length(m))

for j in 1:length(m)
    println("---------- Simulation $j of ", length(m)," | R = ",R[j]," -----------")
    sim = Simulation(MS(), q, n, m[j],
        navg=navg, convergence=:messages, maxiter=maxiter, gamma=gamma,
        Tmax=Tmax, tol=1e-12,
        b=b[j], samegraph=false, samevector=false, randseed=randseed+navg*Tmax*j,
        verbose=true)
    print(sim)
    sims[j] = sim
end

date = Dates.format(now(), "yyyymmdd_HHMM")
save("basic-"*date*".jld", "sims", sims)

plot(sims, title="Mean distortion\nq=$q, n=$n, gamma=$gamma, navg=$navg,
    Tmax=$Tmax", backend=:pyplot, errorbars=true)
ax = gca()
savefig("../../images/basic-"*date, bbox_inches="tight")

print(sims)
