include("../headers.jl")
using UnicodePlots

algo = MS()
const q = 2
randseed = 1001
gamma = 0
navg = 10
b = 5
n = 480
L = 1
nedges = n*2
lambda = [0.0, 1.0]
rho = [0.0, 0.0, 0.5, 0.5]
m = Int(round(nedges*(sum(rho[j]/j for j in eachindex(rho)))))

sim = Simulation(algo, q, n, m, L, nedges, lambda, rho,
    navg=navg, convergence=:decvars, maxiter=Int(1e4), gamma=gamma, nmin=300,
    b=b, samegraph=false, samevector=false, randseed=randseed, verbose = true)

for i in 1:sim.navg
    println("Result: ", (sim.converged[i] ? "converged" : "unconverged"),
        " after ", sim.iterations[i], " iterations")
    println("Final parity check: ", sum(sim.parity[i]))
    myplt = lineplot(sim.codeword[i][1:sim.iterations[i]], canvas=DotCanvas,
    title="Iterations on a codeword, instance n $i", xlabel="Iterations",
    width=80, height = 5)
    show(myplt)
    println("\n")
end
