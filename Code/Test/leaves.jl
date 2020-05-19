# GOAL: observe the effect of leaves

include("../headers.jl")
using UnicodePlots, PyPlot

algo = MS()
const q = 2
randseed = 1001
gamma = 0
navg = 20
L = 1
Tmax = 5

multipliers = [1, 2, 4, 8]
brange = [0:m:5*m for m in multipliers]
nvals = 480*multipliers
m = 280*multipliers
maxiter = Int(2e2)*(multipliers)
convergence_plts = []
distortion_plts = []
distortion_parity_plts = []
convergence_ratio = zeros(length(brange[1]))
distortions = zeros(length(brange[1]))
distortions_parity = zeros(length(brange[1]))
distortions_sd = zeros(length(brange[1]))
for (j,n) in enumerate(nvals)
    println("#######################################################")
    println("                        n = $n")
    println("#######################################################")

    for (i,b) in enumerate(brange[j])
        println("------------------- b=$b ($i of $(length(brange[j])))-------------------")
        sim = Simulation(algo, q, n, m[j],
            navg=navg, convergence=:messages, maxiter=maxiter[j], gamma=gamma,
            Tmax=Tmax, tol=1e-12, b=b, samegraph=false, samevector=false,
            randseed=randseed, verbose=true)
        print(sim)
        convergence_ratio[i] = mean(sim.converged)
        distortions[i] = mean(sim.distortions)
        distortions_parity[i] = mean(sim.distortions[sim.parity .== 0])
        distortions_sd[i] = std(sim.distortions)
    end

    convergence_plt = lineplot(brange[j], convergence_ratio, canvas=DotCanvas,
        title="Effect of removing b factors on different graph, different vector. n=$n",
        xlabel="b",  name="Fraction of converged instances", width=60, height = 10)
    push!(convergence_plts, convergence_plt)

    distortion_plt = lineplot(brange[j], distortions, canvas=DotCanvas,
        title="Effect of removing b factors on different graph, different vector. n=$n",
        xlabel="b", name="Average distortion (0.5 for non-parity)", width=60, height = 10)
    # lineplot!(distortion_plt, brange, distortions_parity, name="Average distortion for instances that fulfilled parity")
    push!(distortion_plts, distortion_plt)

    distortion_parity_plt = lineplot(brange[j], distortions_parity, canvas=DotCanvas,
        title="Effect of removing b factors on different graph, different vector. n=$n",
        xlabel="b", name="Average distortion for instances that fulfilled parity", width=60, height = 10)
    push!(distortion_parity_plts, distortion_parity_plt)
    println()
end

println("Simulation with:
    navg = $navg
    Tmax = $Tmax
    gamma = $gamma
    q = $q
    n: varying
    b: varying
    ")

println("########## CONVERGENCE RATIO ##########")
for myplt in convergence_plts
    println(); show(myplt); println()
end

println("########## AVERAGE DISTORTION ##########")
for myplt in distortion_plts
    println(); show(myplt); println()
end

for myplt in distortion_parity_plts
    println(); show(myplt); println()
end

print("\a")
