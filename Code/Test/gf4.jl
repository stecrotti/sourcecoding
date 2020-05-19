include("../headers.jl")

const q = 4
gamma = 0
b = Int.(round.(500*[0.2, 0.4, 0.6, 0.8]))
n = 420*4
m = Int.(round.(n*[0.2, 0.4, 0.6, 0.8]))
R = 1 .- m/n
L = 1
navg = 20
randseed = 10000
Tmax = 3
D_parity = 0.5*ones(length(m))
D_total = 0.5*ones(length(m))

sims = Vector{Simulation}(undef, length(m))

for j in 1:length(m)
    println("---------- Starting simulation $j of ", length(m)," -----------")
    sim = Simulation(MS(), q, n, m[j],
        navg=navg, convergence=:messages, maxiter=Int(3e2), gamma=gamma, Tmax=Tmax,
        tol=1e-20, b=b[j], samegraph=false, samevector=false, randseed=randseed+navg*Tmax*j,
        verbose=true)
    print(sim)
    D_parity[j] = mean(sim.distortions[sim.parity .== 0])
    D_total[j] = mean(sim.distortions)
    sims[j] = sim
end

plt1 = plotdist(D_parity, R, :unicode, linename="Parity fulfilled")
show(plt1)
println("\n")
plt2 = plotdist(D_total, R, :unicode, linename="Total distortion")
show(plt2)
