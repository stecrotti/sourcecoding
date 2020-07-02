include("../../headers.jl")

const q = 8
const gamma = 1e-2
const n = Int(round(420*2/log2(q)))
const R = collect(0.05:0.05:0.45)
const m = Int.(round.(n*(1 .- R)))
const b = Int(round(n/10))*ones(Int, length(m))
const maxiter = Int(3e2)
const navg = 10
const randseed = 43
const Tmax = 3

sims = Vector{Simulation}(undef, length(m))

for j in 1:length(m)
    println("---------- Simulation $j of ", length(m)," | R = ",R[j]," -----------")
    sim = Simulation(MS(), q, n, m[j], L=2,
        navg=navg, convergence=:parity, maxiter=maxiter, gamma=gamma, Tmax=Tmax,
        b=b[j], samegraph=false, samevector=false, randseed=randseed+navg*Tmax*j,
        verbose=true, arbitrary_mult = true)
    print(sim)
    sims[j] = sim
end

date = Dates.format(now(), "yyyymmdd_HHMM")
PyPlot.close("all")
plot(sims, title="Mean distortion\nq=$q, n=$n, gamma=$gamma, navg=$navg,
    Tmax=$Tmax, arbitrary multiplication", backend=:pyplot, errorbars=true)
ax = gca()
savefig("../../images/gf$q-arbitrary-"*date, bbox_inches="tight")
# save("gf$q-arbitrary.jld", "sims", sims, "date", date)

print(sims)
