include("../headers.jl")

const q = 4
gamma = 1e-3
# b = Int.(round.(500*[0.2, 0.4, 0.6, 0.8]))
# b = 300*ones(Int, 4)
const n = 420*4
const R = collect(0.1:0.1:0.9)
const m = Int.(round.(n*(1 .- R)))
const b = Int.(round.(n*(-R.^2/14 .+ R/7 .+ 1/10)))

navg = 200
randseed = 10000
Tmax = 6

sims = Vector{Simulation}(undef, length(m))

for j in 1:length(m)
    println("---------- Starting simulation $j of ", length(m)," -----------")
    sim = Simulation(MS(), q, n, m[j],
        navg=navg, convergence=:parity, maxiter=Int(5e2), gamma=gamma, Tmax=Tmax,
        tol=1e-20, b=b[j], samegraph=false, samevector=false, randseed=randseed+navg*Tmax*j,
        verbose=true)
    print(sim)
    sims[j] = sim
end

print(sims)
plot(sims)

plot(sims, title="Mean distortion\nq=$q, n=$n, gamma=$gamma, navg=$navg,
    Tmax=$Tmax", backend=:pyplot)
ax = gca()
ax.annotate("b=$(b)", (0,0))
ax.annotate("b=$(maxiter)", (0,0.2))
savefig("../images/gf4", bbox_inches="tight")
# UnicodePlots.scatterplot(R, b, canvas = DotCanvas, xlabel="R", ylabel="b",
    # name="Factors removed")
#
# plot(sims, title="Mean distortion\nq=$q, n=$n, gamma=$gamma, navg=$navg, maxiter=$maxiter,
#     Tmax=$Tmax", backend=:pyplot)
# ax = gca()
# ax.annotate("b=$(reverse(b))", (0,0))
# savefig("../images/parity2", bbox_inches="tight")
