include("../headers.jl")

nsim = 3    # Number of simulations to be run = number of pairs graph,vector
gamma = 1e-6
navg = 10
b = 5
nmin = 300
maxiter = Int(1e4)
algo = MS()

info = ("\n############################ Info ############################
Some simulations with GF(4).
b = $b, gamma=$gamma
##############################################################\n")
const q = 4
n = 480
L = 1
nedges = n*2
lambda = [0.0, 1.0]
rho = [0.0, 0.0, 0.5, 0.5]
m = Int(round(nedges*(sum(rho[j]/j for j in eachindex(rho)))))

sims = Simulation[]
println(info)

for s in 1:nsim
    println("---------------------------------------------")
    println("               Simulation $s     ")
    println("---------------------------------------------")
    sim = Simulation(MS(), q, n, m, L, nedges, lambda, rho,
        navg=navg, convergence=:decvars, maxiter=Int(1e4), gamma=gamma, nmin=500, b=b,
        samegraph=false, samevector=false, verbose = true)
    push!(sims, sim)
    print(stdout, sim, options=:short)
    println()
end
jldopen("gf4.jld", "w") do file
    write(file, "sims", sims)
end
print("\a") # beep
