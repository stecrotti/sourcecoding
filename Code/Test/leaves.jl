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
avgdist_c = [mean(sim.distortions[sim.parity .== 0]) for sim in sims]
sddist_c = [std(sim.distortions[sim.parity .== 0])/sqrt(sim.navg) for sim in sims]
date = Dates.format(now(), "yyyymmdd_HHMM")
save("leaves-"*date*".jld", "sims", sims)

print(sims)
myplt = UnicodePlots.scatterplot(bvals, avgdist, canvas=DotCanvas)

PyPlot.close("all")
PyPlot.figure()
# col1 = "#1f77b4"
col1 = "blue"
PyPlot.errorbar(b, avgdist, sddist, fmt="s", ms=6, capsize=4, color=col1)
ax1 = gca()
ax1.set_xlabel("b")
ax1.set_ylabel("Total distortion", color=col1)
ax1.tick_params(axis="y", labelcolor=col1)

# col2 = "#ff7f0e"
col2 = "red"
ax2 = ax1.twinx()
ax2.errorbar(b, avgdist_c, sddist_c, fmt="v", ms=6, capsize=4, color=col2)
ax2.set_ylabel("Distortion - converged only", color=col2)
ax2.tick_params(axis="y", labelcolor=col2)

plt.tight_layout()
plt.savefig("../images/leaves.pgf")

ratio = [mean(sim.converged) for sim in sims]
PyPlot.close("all")
PyPlot.plot(b,ratio, "o-")
plt.:xlabel("b")
plt.:ylabel("Fraction of converged instances")
plt.tight_layout()
plt.savefig("../images/leavesbars.pgf")
