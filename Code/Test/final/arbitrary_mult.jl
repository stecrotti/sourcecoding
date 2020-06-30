include("../../headers.jl")

const q = 8
const gamma = 1e-4
const n = Int(420*1/log2(q))
const R = collect(0.1:0.1:0.9)
const m = Int.(round.(n*(1 .- R)))
const b = Int(round(n/10))*ones(Int, length(m))
const maxiter = Int(2e2)
const navg = 10
const randseed = 4321
const Tmax = 5

sims = Vector{Simulation}(undef, length(m))

for j in 1:length(m)
    println("---------- Simulation $j of ", length(m)," | R = ",R[j]," -----------")
    sim = Simulation(BP(), q, n, m[j], L=2, nmin=50,
        navg=navg, convergence=:parity, maxiter=maxiter, gamma=gamma, Tmax=Tmax,
        b=b[j], samegraph=true, samevector=false, randseed=randseed+navg*Tmax*j,
        verbose=true, arbitrary_mult = false)
    print(sim)
    sims[j] = sim
end

# date = string(Dates.today())
# plot(sims, title="Mean distortion\nq=$q, n=$n, gamma=$gamma, navg=$navg,
#     Tmax=$Tmax", backend=:pyplot, errorbars=true)
# ax = gca()
# savefig("../../images/gf$q-arbitrary-"*date, bbox_inches="tight")
# save("gf$q-arbitrary.jld", "sims", sims, "date", date)

print(sims)
plot(sims)
