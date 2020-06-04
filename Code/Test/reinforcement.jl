include("../headers.jl")

const q = 2
# b = Int.(round.(400*sqrt.([0.2, 0.4, 0.6, 0.8])))
# b = Int.(round.(350*([0.2, 0.4, 0.6, 0.8]).^0.2))

const n = 420*6
const R = collect(0.1:0.1:0.9)
const m = Int.(round.(n*(1 .- R)))
# b = Int.(round.(n/6))*ones(Int, length(m))
# b = Int.(round.(n/3*(0.1 .+ R .* (1 .- R))))
# b = Int.(round.(n*(-R.^2/14 .+ R/7 .+ 1/10)))
b = 100*ones(Int, length(R))

navg = 300
randseed = 99
Tmax = 8

# gamma = 1e-2
# maxiter = Int(round(1/gamma))
# nmin = Int(round(0.6/maxiter))
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
#     sims2[j] = sim
# end
#
# print(sims2)
# plot(sims2)


#############################################################################
# gamma = 1e-3
gamma = 1e-2
# maxiter = Int(round(1/gamma))
maxiter = 500*ones(Int, length(R))
nmin = Int.(round.(0.7*maxiter))

println("#######################################")
println("#              γ = $gamma              #")
println("#######################################")

sims = Vector{Simulation}(undef, length(m))

for j in 1:length(m)
    println("---------- Starting simulation $j of ", length(m)," -----------")
    sim = Simulation(MS(), q, n, m[j],
        navg=navg, convergence=:parity, maxiter=maxiter[j], gamma=gamma, Tmax=Tmax,
        nmin=nmin[j], b=b[j], samegraph=false, samevector=false,
        randseed=randseed+navg*Tmax*j, verbose=true)
    print(sim)
    sims[j] = sim
end

print(sims)

plot(sims, title="Mean distortion\nq=$q, n=$n, gamma=$gamma, navg=$navg,
    Tmax=$Tmax", backend=:pyplot)
ax = gca()
ax.annotate("b=$(b)", (0,0), fontsize="small")
ax.annotate("maxiter=$(maxiter)", (0,0.04), fontsize="small")
savefig("../images/parity1e2", bbox_inches="tight")

plot(sims)

# UnicodePlots.scatterplot(R, b, canvas = DotCanvas, xlabel="R", ylabel="b",
    # name="Factors removed")

############################
# gamma = 1e-4
# maxiter = Int(round(1/gamma))
# nmin = Int(round(0.7*maxiter))
#
# println("#######################################")
# println("#              γ = $gamma             #")
# println("#######################################")
#
# sims4 = Vector{Simulation}(undef, length(m))
#
# for j in 1:length(m)
#     println("---------- Starting simulation $j of ", length(m)," -----------")
#     sim = Simulation(MS(), q, n, m[j],
#         navg=navg, convergence=:decvars, maxiter=maxiter, gamma=gamma, Tmax=Tmax,
#         nmin=nmin, b=b[j], samegraph=false, samevector=false,
#         randseed=randseed+navg*Tmax*j, verbose=true)
#     print(sim)
#     sims4[j] = sim
# end
#
# print(sims4)
# plot(sims4)
