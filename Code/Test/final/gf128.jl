include("../../headers.jl")

const q = 128
const gamma = 5e-3
const n = Int(420*6/log2(q))
const R = collect(0.1:0.1:0.9)
const m = Int.(round.(n*(1 .- R)))
const b = Int(round(n/50))*ones(Int, length(m))
const maxiter = Int(1e3)
const navg = 50
const randseed = 100
const Tmax = 6

sims = Vector{Simulation}(undef, length(m))

for j in 1:length(m)
    println("---------- Simulation $j of ", length(m)," | R = ",R[j]," -----------")
    sim = Simulation(MS(), q, n, m[j],
        navg=navg, convergence=:parity, maxiter=maxiter, gamma=gamma, Tmax=Tmax,
        b=b[j], samegraph=false, samevector=false, randseed=randseed+navg*Tmax*j,
        verbose=true)
    print(sim)
    sims[j] = sim
end

date = string(Dates.today())
PyPlot.close("all")
plot(sims, title="Mean distortion\nq=$q, n=$n, gamma=$gamma, navg=$navg,
    Tmax=$Tmax", backend=:pyplot, errorbars=true)
ax = gca()
ax.annotate("b=$(b)", (0,0))
ax.annotate("maxiter=$(maxiter)", (0,0.05))
savefig("../../images/gf$q-"*date, bbox_inches="tight")
save("gf$q.jld", "sims", sims, "date", date)

print(sims)
plot(sims)
