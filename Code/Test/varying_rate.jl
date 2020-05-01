include("../headers.jl")

const q = 2
gamma = 0
b = Int.(round.(10*[0.2, 0.4, 0.6, 0.8]))
n = 420
m = Int.(round.(n*[0.2, 0.4, 0.6, 0.8]))
L = 1
navg = 20
nedges = n*2
randseed = 8780
Tmax = 4
D_parity = zeros(length(m))
D_total = zeros(length(m))

sims = Vector{Simulation}(undef, length(m))

for j in 1:length(m)
    println("---------- Starting simulation $j of ", length(m)," -----------")
    sim = Simulation(MS(), q, n, m[j],
        navg=navg, convergence=:messages, maxiter=200, gamma=gamma, Tmax=Tmax,
        tol=1e-12, b=b[j], samegraph=false, samevector=false, randseed=randseed+navg*Tmax*j,
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

# ############# q = 4 #################
# q = 4
# D_converged4 = zeros(length(m))
# D_total4 = zeros(length(m))
# R4 = zeros(length(m))
#
# for j in 1:length(m)
#     println("---------- Starting simulation $j of ", length(m)," -----------")
#     sim = Simulation(MS(), q, n, m[j], L, nedges, lambda[j], rho[j],
#         navg=navg, convergence=:decvars, maxiter=Int(1e3), gamma=gamma, nmin=300,
#         b=b, samegraph=false, samevector=false, randseed=randseed, verbose = true)
#     print(sim)
#     R4[j] = sim.R
#     D_converged4[j] = meandist(sim, convergedonly=true)
#     D_total4[j] = meandist(sim, convergedonly=false)
# end
#
# scatterplot!(myplt, D_converged4, R4, name="GF4")
