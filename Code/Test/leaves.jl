# GOAL: observe the effect of leaves

include("../headers.jl")

const q = 2
const gamma = 0
const n = Int(420*2/log2(q))
const R = 0.7
const m = Int.(round.(n*(1 .- R)))
const navg = 100
const randseed = 99
const Tmax = 3
const maxiter = Int(1e3)

bvals = [0, 1, 5, 15, 30]
sims = Vector{Simulation}(undef, length(bvals))

for (j,b) in enumerate(bvals)
    println("\n---------- Simulation $j of ", length(bvals)," | b = ",bvals[j]," -----------")
    sims[j] = Simulation(MS(), q, n, m,
        navg=navg, convergence=:parity, maxiter=maxiter, gamma=gamma, Tmax=Tmax,
        b=b, samegraph=false, samevector=false, randseed=randseed+navg*Tmax*j,
        verbose=true)
end

avgdist = [mean(sim.distortions) for sim in sims]
sddist = [std(sim.distortions)/sqrt(sim.navg) for sim in sims]
date = Dates.format(now(), "yyyymmdd_HHMM")
# save("leaves-"*date*".jld", "sims", sims)

print(sims)
myplt = UnicodePlots.scatterplot(bvals, avgdist, canvas=DotCanvas)
