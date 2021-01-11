include("./../headers.jl")

algo = OptimalCycle()

nn = Int.(round.(1000:150:3000))
R = 0.5
mm = Int.(round.(nn .* (1-R))) 
navg = 100
runtimes = [zeros(navg) for n in nn]

for (i,n) in enumerate(nn)
    m = mm[i]
    sim = Simulation(2, n, m, algo, niter=navg, verbose=false)
    runtimes[i] .= sim.runtimes
    println("####### Size $i of $(length(nn)) finished.")
end

avg_runtime = [mean(r) for r in runtimes]
sd_runtime = [std(r)/sqrt(length(r)) for r in runtimes]

pl = plot(nn, avg_runtime, yerror=sd_runtime, marker=:circle, label="Runtimes",
    legend=:outertopright)
xlabel!("n")
smooth_range = LinRange(minimum(nn), maximum(nn), 100)
pow = 2;
plot!(pl, smooth_range, 1.2e-7*smooth_range.^pow, label="0(n^$pow)")