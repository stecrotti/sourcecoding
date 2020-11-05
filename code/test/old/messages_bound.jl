# GOAL: see if message differences are bounded

include("../headers.jl")
using UnicodePlots

algo = MS()
const q = 2
randseed = 1001
gamma = 0
navg = 5
b = 5
n = 480
L = 1
nedges = n*2
lambda = [0.0, 1.0]
rho = [0.0, 0.0, 0.5, 0.5]
m = Int(round(nedges*(sum(rho[j]/j for j in eachindex(rho)))))

sim = Simulation(algo, q, n, m, L, nedges, lambda, rho,
    navg=navg, convergence=:messages, maxiter=Int(1e3), gamma=gamma, tol=1e-7,
    b=b, samegraph=true, samevector=true, randseed=randseed, verbose=true)

print(sim)

println(sim.maxchange)

# for i in 1:sim.navg
#     println("Result: ", (sim.converged[i] ? "converged" : "unconverged"),
#         " after ", sim.iterations[i], " iterations")
#     println("Parity check: ", sum(sim.parity[i]))
#     myplt = lineplot(sim.maxdiff[i][1:sim.iterations[i]], canvas=DotCanvas,
#     title="Max absolute difference in messages, instance n $i", xlabel="Iterations",
#     width=80, height = 5)
#     show(myplt)
#     println("\n")
# end

print("\a")
