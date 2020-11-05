include("../headers.jl")
PyPlot.close("all")

nsim = 20    # Number of simulations to be run = number of pairs graph,vector
gamma = 1e-6
navg = 50
b = 2
Tmax = 5

info = ("\n############################ Info ############################
Some simulations with same graph, same input vector, Tmax = $Tmax, gamma=$gamma, b=$b.
GOAL: find out whether there are some configurations {graph,vector} that
 give bad results no matter how many randomized repetitions
##############################################################\n")
const q = 2
n = 480
L = 1
nedges = n*2
lambda = [0.0, 1.0]
rho = [0.0, 0.0, 0.5, 0.5]
m = Int(round(nedges*(sum(rho[j]/j for j in eachindex(rho))))) - b
sims = Simulation[]

println(info)

for s in 1:nsim
    println("---------------------------------------------")
    println("               Simulation $s     ")
    println("---------------------------------------------")
    sim = Simulation(MS(), q, n, m, L,
        navg=navg, convergence=:decvars, maxiter=Int(1e4), gamma=gamma, nmin=300,
        b=b, samegraph=true, samevector=true, verbose = false)
    push!(sims, sim)
    print(stdout, sim, options=:short)
    println()
end
jldopen("same_graph_gamma1e-6.jld", "w") do file
    write(file, "sims", sims)
end
print("\a") # beep
