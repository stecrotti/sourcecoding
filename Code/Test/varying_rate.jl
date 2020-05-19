include("../headers.jl")

const q = 2
gamma = 0
# b = Int.(round.(400*[0.2, 0.4, 0.6, 0.8]))
 b = Int.(round.(400*sqrt.([0.2, 0.4, 0.6, 0.8])))
n = 420*4
m = Int.(round.(n*[0.2, 0.4, 0.6, 0.8]))
R = 1 .- m/n
L = 1
navg = 50
nedges = n*2
randseed = 8780
Tmax = 5
D_parity = zeros(length(m))
D_total = zeros(length(m))

sims = Vector{Simulation}(undef, length(m))

for j in 1:length(m)
    println("---------- Starting simulation $j of ", length(m)," -----------")
    sim = Simulation(MS(), q, n, m[j],
        navg=navg, convergence=:messages, maxiter=300, gamma=gamma, Tmax=Tmax,
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
