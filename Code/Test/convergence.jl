# GOAL: observe convergence of messages

include("../headers.jl")
using UnicodePlots, PyPlot

algo = MS()
const q = 2
randseed = 100
gamma = 0
navg = 20
maxiter = Int(1e3)
b = 5
n = 480
m = 280
L = 0.1
Tmax = 5
nsims = 10

println("#### $nsims simulations with n=$n, b=$b, Tmax=$Tmax, maxiter=$maxiter,",
    " navg=$navg. Different vector, same graph",
    "\nGoal: see if there are some instances that never work ####\n")

sims = Vector{Simulation}(undef, nsims)
for s in 1:nsims
    println("---- Simulation $s of $nsims ----")
    sims[s] = Simulation(algo, q, n, m,
        navg=navg, convergence=:messages, maxiter=maxiter, gamma=gamma, Tmax=Tmax,
        tol=1e-7, b=b, samegraph=true, samevector=false, randseed=randseed+navg*Tmax*s,
        verbose=false)
    print(sims[s])
end

seeds = string.([randseed+navg*Tmax*s for s in eachindex(sims)])
avgtrials = [mean_sd_string(sim.trials[sim.converged .== true]) for sim in sims]
convergence_ratio = [mean_sd_string(sim.converged) for sim in sims]
avgdist = [mean_sd_string(sim.distortions) for sim in sims]
avgiters = [mean_sd_string(sim.iterations[sim.converged .== true]) for sim in sims]
data = hcat(seeds, convergence_ratio, avgdist, avgiters, avgtrials)

println("#### $nsims simulations with n=$n, R=$(1-m/n) b=$b, Tmax=$Tmax, maxiter=$maxiter,",
    " navg=$navg. Different vector, same graph",
    "\nGoal: see if there are some instances that never work ####\n")
pretty_table(data, ["Seed" "Convergence ratio" "Distortion" "Iterations (converged)" "Trials (converged)"],
    alignment=:c)


# println("### $nsim simulations with n=$n seed=$randseed, b=$b, Tmax=$Tmax, ",
#     "maxiter=$maxiter ###\n")
#
# sim = Simulation(algo, q, n, m, L, nedges, lambda, rho,
#     navg=navg, convergence=:messages, maxiter=maxiter, gamma=gamma, Tmax=Tmax,
#     tol=1e-7, b=b, samegraph=true, samevector=true, randseed=randseed+navg, verbose=true)

print("\a")
