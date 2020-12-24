include("./../headers.jl")
using JLD2

q = 2
gamma = 5e-3
n = Int(round(420*3/log2(q)))
R = collect(0.21:0.1:0.81) 
m = Int.(round.(n*(1 .- R)))
b = Int(round(n/100))*ones(Int, length(m))
maxiter = Int(2e2)
niter = 10
randseed = 1234
Tmax = 1
ms = MS(maxiter=maxiter, gamma=gamma, Tmax=Tmax)

sims_cycles = Vector{Simulation{OptimalCycle}}(undef, length(m))
sims_ms = Vector{Simulation{MS}}(undef, length(m))


for j in 1:length(m)
    println("\n---------- Simulation $j of ", length(m)," | R = ",R[j]," -----------")
    # sims_ms[j] = Simulation(q, n, m[j], ms, niter=niter, b=b[j], 
    #     randseed=randseed, showprogress=true)
    sims_cycles[j] = Simulation(q, n, m[j], OptimalCycle(), b=0, niter=niter,
        randseed=randseed, showprogress=true)
end

# JLD2.@save "./rdb_optimal_cycles2.jld" sims_cycles sims_ms

# include("./../headers.jl")
# using JLD2

# q = 2
# N = [0.1,0.5,5]
# n = [Int(round(420*nn/log2(q))) for nn in N]
# R = collect(0.21:0.1:0.81) 
# m = [Int.(round.(nn*(1 .- R))) for nn in n]
# niter = 10
# randseed = 1234

# algo = OptimalCycle()

# sims = [Vector{Simulation{OptimalCycle}}(undef, length(m[1])) for nn in n]

# for i in eachindex(n)
#     println("#### size $i of $(length(n)) | n = $(n[i]) ####")
#     for j in 1:length(m[i])
#         println("\n---------- Simulation $j of ", length(m[i])," | R = ",R[j]," -----------")
#         sim = Simulation(q, n[i], m[i][j], algo, niter=niter, b=0, randseed=randseed, 
#             verbose=false)
#         sims[i][j] = sim
#     end
# end

# JLD2.@save "./rdb_optimal_cycles1.jld" sims

# pl = plot()
# for sim in sims
#     plot!(pl, sim, label="n = $(sim[1].n)", ms=2)
# end
# display(pl)
