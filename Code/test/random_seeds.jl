include("../headers.jl")

randseed = 9991
gamma = 1e-6
navg = 3
b = 5
const q = 2
n = 480
L = 1
nedges = n*2
lambda = [0.0, 1.0]
rho = [0.0, 0.0, 0.5, 0.5]
m = Int(round(nedges*(sum(rho[j]/j for j in eachindex(rho)))))

sim1 = Simulation(MS(), q, n, m, L, nedges, lambda, rho,
    navg=navg, convergence=:decvars, maxiter=Int(1e4), gamma=gamma, nmin=300,
    b=b, samegraph=false, samevector=false, randseed=randseed, verbose = true)

sim2 = Simulation(MS(), q, n, m, L, nedges, lambda, rho,
    navg=navg, convergence=:decvars, maxiter=Int(1e3), gamma=gamma, nmin=300,
    b=b, samegraph=false, samevector=false, randseed=randseed, verbose = true)
println("These two should be equal:")
[sim1.iterations sim2.iterations]
