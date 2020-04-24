# GOAL: observe convergence of messages

include("../headers.jl")
using UnicodePlots, PyPlot

algo = MS()
const q = 2
randseed = 10010
gamma = 0
navg = 10
maxiter = Int(1e3)
b = 50
n = 480
L = 1
Tmax = 5
nedges = n*2
lambda = [0.0, 1.0]
rho = [0.0, 0.0, 0.5, 0.5]
m = Int(round(nedges*(sum(rho[j]/j for j in eachindex(rho))))) - b
nsim = 10

println("### $nsim simulations with n=$n seed=$randseed, b=$b, Tmax=$Tmax, ",
    "maxiter=$maxiter ###\n")
#
# sims = Simulation[]
#
# for rs in 1:nsim
#     println("-- Starting simulation $rs of $nsim...")
#     sim = Simulation(algo, q, n, m, L, nedges, lambda, rho,
#         navg=navg, convergence=:messages, maxiter=maxiter, gamma=gamma, Tmax=Tmax,
#         tol=1e-7, b=b, samegraph=true, samevector=true, randseed=randseed+rs*navg,
#         verbose=true)
#     push!(sims,sim)
# end
# for rs in 1:nsim
#     print(sims[rs])
# end
#
# Tmax = 5
# maxiter = Int(1e4)
# println("### Take the graph and vector from the first of the previous simulations",
#  " and try running it for longer: Tmax=$Tmax, maxiter=$maxiter ###\n")
println("-- Starting simulation...")
sim = Simulation(algo, q, n, m, L, nedges, lambda, rho,
    navg=navg, convergence=:messages, maxiter=maxiter, gamma=gamma, Tmax=Tmax,
    tol=1e-7, b=b, samegraph=true, samevector=true, randseed=randseed+navg, verbose=true)

# print(sim)
# PyPlot.close("all")
# fig = plot(sim)
# plt.savefig("trial.png")

# for i in 1:sim.navg
#     println("Result: ", (sim.converged[i] ? "converged" : "unconverged"),
#         " after ", sim.iterations[i], " iterations")
#     println("Parity check: ", sum(sim.parity[i]))
#     myplt = lineplot(sim.maxchange[i][1:sim.iterations[i]], canvas=DotCanvas,
#     title="Max difference in messages wrt previous iter, instance n $i", xlabel="Iterations",
#     width=80, height = 5)
#     show(myplt)
#     println("\n")
# end

print("\a")
