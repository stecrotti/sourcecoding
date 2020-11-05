include("../headers.jl")

algo = MS()
const q = 2
randseed = 100
gamma = 0
maxiter = Int(5e3)
b = 3
n = 480
m = 280
L = 1
Tmax = 5
navg = 30
tol = 1e-22

alphas = 0:0.2:1
nsims = length(alphas)
sims = Vector{Simulation}(undef, nsims)

for (s,alpha) in enumerate(alphas)
    println("\n----- Simulation with α=$alpha ($s of $nsims) -----")
    sims[s] = Simulation(algo, q, n, m,
        navg=navg, convergence=:messages, maxiter=maxiter, gamma=gamma, alpha=alpha,
        Tmax=Tmax, tol=tol, b=b, samegraph=true, samevector=false,
        randseed=randseed+navg*Tmax, verbose=true)
    print(sims[s])
end

println("###### Test damping. Tolerance for messages convergence: $tol, maxiter:$maxiter ######\n")

for (s,alpha) in enumerate(alphas)
    println("----- α=$alpha -----")
    print(sims[s], "\n")
end

println("\a")
