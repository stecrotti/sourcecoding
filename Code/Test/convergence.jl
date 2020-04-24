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

println("### $nsim simulations with n=$n seed=$randseed, b=$b, Tmax=$Tmax, ",
    "maxiter=$maxiter ###\n")

sim = Simulation(algo, q, n, m, L, nedges, lambda, rho,
    navg=navg, convergence=:messages, maxiter=maxiter, gamma=gamma, Tmax=Tmax,
    tol=1e-7, b=b, samegraph=true, samevector=true, randseed=randseed+navg, verbose=true)

print("\a")
