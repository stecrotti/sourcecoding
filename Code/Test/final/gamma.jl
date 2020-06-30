include("../../headers.jl")

const q = 64
const n = Int(420*6/log2(q))
const R = 0.6
const m = Int(round(n*(1 - R)))
const b = Int(round(n/50))
const maxiter = Int(5e2)
const navg = 200
const randseed = 100
const Tmax = 5

gamma_vals = 10 .^ LinRange(-2.5,-1.5,7)

sims = Vector{Simulation}(undef, length(gamma_vals))

for (j,gamma) in enumerate(gamma_vals)
    println("\n---------- Simulation $j of ", length(gamma_vals),
                " | γ = ",gamma," -----------")
    sims[j] = Simulation(MS(), q, n, m,
        navg=navg, convergence=:parity, maxiter=maxiter, gamma=gamma, Tmax=Tmax,
        b=b, samegraph=false, samevector=false, randseed=randseed+navg*Tmax*j,
        verbose=true)
    print(sims[j])
end

date = Dates.format(now(), "yyyymmdd_HHMM")

avgiters = [mean(sim.iterations[sim.converged]) for sim in sims]
avgtrials = [mean(sim.trials[sim.converged]) for sim in sims]
avgdist = [meandist(sim, convergedonly=false) for sim in sims]
avgdistconverged = [meandist(sim, convergedonly=true) for sim in sims]
avgconvergence, sdconvergence = convergence_prob(sims)

convplot = UnicodePlots.lineplot(gamma_vals, avgconvergence, canvas=DotCanvas,
    title="Average convergence", xlabel="gamma")
itersplot = UnicodePlots.lineplot(gamma_vals, avgiters, canvas=DotCanvas,
    title="Average iterations for converged instances", xlabel="gamma")
trialsplot = UnicodePlots.lineplot(gamma_vals, avgtrials, canvas=DotCanvas,
    title="Average trials for converged instances", xlabel="gamma")
avgdistplot = UnicodePlots.lineplot(gamma_vals, avgdist, canvas=DotCanvas,
    title="Average distortion", xlabel="gamma")
avgdistconvergedplot = UnicodePlots.lineplot(gamma_vals, avgdistconverged, canvas=DotCanvas,
    title="Average distortion for converged instances", xlabel="gamma")
# plot(sims, title="Mean distortion\nq=$q, n=$n, gamma=$gamma, navg=$navg,
#     Tmax=$Tmax", backend=:pyplot, errorbars=true)
# ax = gca()
# ax.annotate("b=$(b)", (0,0))
# ax.annotate("maxiter=$(maxiter)", (0,0.05))
# savefig("../../images/gamma-"*date, bbox_inches="tight")
save("gamma"*date*".jld", "sims", sims, "date", date, "gamma_vals", gamma_vals)

print(sims)
display(convplot)
display(itersplot)
display(trialsplot)
display(avgdistplot)
display(avgdistconvergedplot)
