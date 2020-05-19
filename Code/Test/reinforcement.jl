include("../headers.jl")

const q = 2
# gamma = 1e-2
# maxiter = Int(round(1/gamma))
# nmin = Int(round(0.6/maxiter))
# # b = Int.(round.(400*[0.2, 0.4, 0.6, 0.8]))
b = Int.(round.(400*sqrt.([0.2, 0.4, 0.6, 0.8])))
n = 420*4
m = Int.(round.(n*[0.2, 0.4, 0.6, 0.8]))
R = 1 .- m/n
navg = 30
randseed = 8780
Tmax = 3
D_parity = zeros(length(m))
D_total = zeros(length(m))
#
# sims2 = Vector{Simulation}(undef, length(m))
#
# println("#######################################")
# println("#              γ = $gamma               #")
# println("#######################################")
#
# for j in 1:length(m)
#     println("---------- Starting simulation $j of ", length(m)," -----------")
#     sim = Simulation(MS(), q, n, m[j],
#         navg=navg, convergence=:decvars, maxiter=maxiter, gamma=gamma, Tmax=Tmax,
#         nmin=nmin, b=b[j], samegraph=false, samevector=false,
#         randseed=randseed+navg*Tmax*j, verbose=true)
#     print(sim)
#     D_parity[j] = mean(sim.distortions[sim.parity .== 0])
#     D_total[j] = mean(sim.distortions)
#     sims2[j] = sim
# end
#
# print(sims2)
# plot(sims2)
#

#############################################################################
# gamma = 1e-3
# maxiter = Int(round(1/gamma))
# nmin = Int(round(0.7*maxiter))
#
# println("#######################################")
# println("#              γ = $gamma              #")
# println("#######################################")
#
# sims3 = Vector{Simulation}(undef, length(m))
#
# for j in 1:length(m)
#     println("---------- Starting simulation $j of ", length(m)," -----------")
#     sim = Simulation(MS(), q, n, m[j],
#         navg=navg, convergence=:decvars, maxiter=maxiter, gamma=gamma, Tmax=Tmax,
#         nmin=nmin, b=b[j], samegraph=false, samevector=false,
#         randseed=randseed+navg*Tmax*j, verbose=true)
#     print(sim)
#     D_parity[j] = mean(sim.distortions[sim.parity .== 0])
#     D_total[j] = mean(sim.distortions)
#     sims3[j] = sim
# end
#
# print(sims3)
# plot(sims3)


gamma = 1e-4
maxiter = Int(round(1/gamma))
nmin = Int(round(0.7*maxiter))

println("#######################################")
println("#              γ = $gamma             #")
println("#######################################")

sims4 = Vector{Simulation}(undef, length(m))

for j in 1:length(m)
    println("---------- Starting simulation $j of ", length(m)," -----------")
    sim = Simulation(MS(), q, n, m[j],
        navg=navg, convergence=:decvars, maxiter=maxiter, gamma=gamma, Tmax=Tmax,
        nmin=nmin, b=b[j], samegraph=false, samevector=false,
        randseed=randseed+navg*Tmax*j, verbose=true)
    print(sim)
    D_parity[j] = mean(sim.distortions[sim.parity .== 0])
    D_total[j] = mean(sim.distortions)
    sims4[j] = sim
end

print(sims4)
plot(sims4)
