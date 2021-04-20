using OffsetArrays, Statistics, DelimitedFiles
include("../bp_full.jl")
include("../slim_graphs.jl")

m = factorial(5)*11*4
R = 0.8/3
r = 1 - 3R
n = round(Int, 3m/(2+r))
Λ = OffsetVector([0,0,1-r,r], 0:3)
K = OffsetVector([0,0,0,1], 0:3)
nedges = 3m
Hs = [0.65, 0.8, 0.95]
navg = 10
dist_f3 = [Float64[] for _ in eachindex(Hs)]
for j in 1:navg
    println("#### Round $j of $navg ####")
    bp = bp_full(n, m, nedges, Λ, K)
    B, freevars = findbasis_slow(Array(bp.H))
    s = rand((-1,1), n)
    for (i,H) in enumerate(Hs)
        println("------ H=", round(H,digits=4), ". ", i, " of ", length(Hs), " ------")
        efield = [(exp(ss*H),exp(-ss*H)) for ss in s]
        _,_,d = decimate!(bp, efield, freevars, s, maxiter=1000, Tmax=1, tol=1e-4)  
        isnan(d) || push!(dist_f3[i], d) 
    end
end

writedlm("Hs.txt", Hs, ",")
writedlm("dist.txt", dist_f3, ",")