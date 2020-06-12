include("../headers.jl")

const q = 32
gamma = 1e-2
const n = 420*2
const R = collect(0.1:0.1:0.9)
const m = Int.(round.(n*(1 .- R)))
# const b = Int.(round.(n/2*(-R.^2/14 .+ R/7 .+ 1/10)))
const b = Int(round(n/15))*ones(Int, length(m))
maxiter = Int(3e2)
navg = 200
randseed = 10000
Tmax = 6

sims = Vector{Simulation}(undef, length(m))

for j in 1:length(m)
    println("---------- Simulation $j of ", length(m)," | R = ",R[j]," -----------")
    sim = Simulation(MS(), q, n, m[j],
        navg=navg, convergence=:parity, maxiter=maxiter, gamma=gamma, Tmax=Tmax,
        tol=1e-20, b=b[j], samegraph=false, samevector=false, randseed=randseed+navg*Tmax*j,
        verbose=true)
    print(sim)
    sims[j] = sim
end

print(sims)
plot(sims)

plot(sims, title="Mean distortion\nq=$q, n=$n, gamma=$gamma, navg=$navg,
    Tmax=$Tmax", backend=:pyplot, errorbars=true)
ax = gca()
ax.annotate("b=$(b)", (0,0))
ax.annotate("maxiter=$(maxiter)", (0,0.05))
date = string(Dates.today())
savefig("../images/gf32-"*date, bbox_inches="tight")
