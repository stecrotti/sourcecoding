include("../../headers.jl")

# load files
filenames = [fn for fn in readdir() if (fn[end-2:end]==".jl" && fn[1:2]=="gf")]

simsvec = [Simulation[] for fn in filenames]
for (i,fn) in enumerate(filenames)
    simsvec[i] = load(filename, "sims")
end

sort!(simsvec, by = sims -> sims[1].q)

plot(simsvec, title="Mean distortion\n $(simsvec[1][1].n) bits, gamma=$gamma, navg=$navg,
    Tmax=$Tmax", backend=:pyplot, errorbars=false)
ax = gca()
# ax.annotate("b=$( Int(round(n/30))*ones(Int, length(m)))", (0,0))
# ax.annotate("maxiter=$(maxiter)", (0,0.05))
date = string(Dates.today())
savefig("../images/gfq-"*date, bbox_inches="tight")
