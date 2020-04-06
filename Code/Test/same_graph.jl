info = ("\n#################### Info ####################
Some simulations with same graph, same input vector, gamma non-zero.
GOAL: find out whether there are some configurations {graph,vector} that
 give bad distortions no matter how many trials
##############################################\n")

include("../headers.jl")
PyPlot.close("all")

# Number of simulations to be run = number of pairs graph,vector
nsim = 2
sims = Simulation[]

gamma = 1e-4
const q = 2
n = 480
L = 1
nedges = n*2
lambda = [0.0, 1.0]
rho = [0.0, 0.0, 0.5, 0.5]
m = Int(round(nedges*(sum(rho[j]/j for j in eachindex(rho)))))

println(info)

for s in 1:nsim
    println("---------------------------------------------")
    println("               Simulation $s     ")
    println("---------------------------------------------")
    sim = Simulation(MS(), q, n, m, L, nedges, lambda, rho,
        navg=1, convergence=:decvars, maxiter=Int(1e4), gamma=gamma, nmin=300, b=2,
        samegraph=true, samevector=true, verbose = false)
    push!(sims, sim)
    print(stdout, sim, options=:short)
    println()
end
jldopen("same_graph_simulations.jld", "w") do file
    write(file, "sims", sims)
end
print("\a") # beep
