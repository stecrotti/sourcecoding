include("../../headers.jl")

# LOAD FILES
# Regular expression for valid file names
r = r"gf[0-9]+[-][0-9]+_[0-9]+[.]jld"
filenames = [fn for fn in readdir() if match(r,fn)!=nothing]

simsvec = [Simulation[] for fn in filenames]
for (i,fn) in enumerate(filenames)
    simsvec[i] = load(fn, "sims")
end

println("\nFiles loaded\n")

sort!(simsvec, by = sims -> sims[1].q)

println("Plotting image. This might take a while...")

plot(simsvec, title="Mean distortion\n $(simsvec[1][1].n) bits",
        backend=:pyplot, errorbars=true)
ax = gca()
# ax.annotate("b=$( Int(round(n/30))*ones(Int, length(m)))", (0,0))
# ax.annotate("maxiter=$(maxiter)", (0,0.05))
date = Dates.format(now(), "yyyymmdd_HHMM")
# savefig("../../images/gfq-"*date, bbox_inches="tight")
println("Done!")
print("\a")
