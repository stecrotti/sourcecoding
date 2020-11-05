include("headers.jl")

d = load("./Test/final/gamma20200709_0959.jld")
sims = d["sims"]
gamma_vals = d["gamma_vals"]

avgiters = [mean(sim.iterations[sim.converged]) for sim in sims]
avgtrials = [mean(sim.trials[sim.converged]) for sim in sims]
avgdist = [meandist(sim, convergedonly=false) for sim in sims]
avgdistconverged = [meandist(sim, convergedonly=true) for sim in sims]
convergratio = [mean(sim.converged) for sim in sims]

plt.close("all")
pgfsettings()
increasefontsize(2)

plt.figure("avgiters")
plt.semilogx(gamma_vals, avgiters, "o-")
plt.:xlabel("gamma"); plt.:ylabel("Iterations")
ax = gca()
plt.xticks([5e-3, 1e-2, 2e-2], string.([5e-3, 1e-2, 2e-2]))
tight_layout()
savefig("./images/gamma_iters.pgf")

plt.figure("avgdist")
plt.plot(gamma_vals, avgdist, "o-", label="Distortion")
R = sims[1].R
d = rdbinv(R)
plt.axhline(y=d, label="RDB", color="#ff7f0e")
plt.:xlabel("gamma"); # plt.:ylabel("Distortion")
plt.legend()
ax = gca()
ax.set_xscale("log")
plt.xticks([5e-3, 1e-2, 2e-2], string.([5e-3, 1e-2, 2e-2]))
plt.ylim((0.07, 0.14))
tight_layout()
savefig("./images/gamma_dist.pgf")

plt.figure("convergedonly")
plt.semilogx(gamma_vals, avgdistconverged, "o-", label="Distortion")
plt.axhline(y=d, label="RDB", color="#ff7f0e")
plt.:xlabel("gamma");
plt.ylim((0.07, 0.12))
plt.legend()
ax = gca()
plt.xticks([5e-3, 1e-2, 2e-2], string.([5e-3, 1e-2, 2e-2]))
tight_layout()
savefig("./images/gamma_dist2.pgf")

plt.figure("convergence")
plt.semilogx(gamma_vals, convergratio, "o-")
plt.:xlabel("gamma"); plt.:ylabel("Convergence ratio")
ax = gca()
plt.xticks([5e-3, 1e-2, 2e-2], string.([5e-3, 1e-2, 2e-2]))
tight_layout()
savefig("./images/gamma_ratio.pgf")
