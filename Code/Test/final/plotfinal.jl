include("../../headers.jl")

# LOAD FILES
# Regular expression for valid file names
r = r"gf[0-9]+[-][0-9]+_[0-9]+[.]jld"
filenames = [fn for fn in readdir() if match(r,fn)!=nothing]

simsvec = [Simulation[] for fn in filenames]
for (i,fn) in enumerate(filenames)
    simsvec[i] = load(fn, "sims")
end

println("\n $(length(simsvec)) files loaded\n")

sort!(simsvec, by = sims -> sims[1].q)

println("Plotting image. This might take a while...")

pgfsettings("sans-serif")
increasefontsize(1.5)

plt.close("all")
plot(simsvec, title="",
        backend=:pyplot, errorbars=true)
d = load("gf128final.jld")
sd128 = d["sd128"]
avg128 = d["avg128"]
ax = gca()
R = [sim.R for sim in simsvec[1]]
ax.errorbar(R, avg128, sd128, fmt=">", ms=4, capsize=4, label="GF(128)")
plt.legend()
plt.tight_layout()                       
# ax.annotate("b=$( Int(round(n/30))*ones(Int, length(m)))", (0,0))
# ax.annotate("maxiter=$(maxiter)", (0,0.05))
date = Dates.format(now(), "yyyymmdd_HHMM")
plt.savefig("../../../../Latex/images/gfq2.pgf", bbox_inches="tight")
