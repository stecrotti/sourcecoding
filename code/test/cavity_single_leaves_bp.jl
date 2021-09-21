include("../bp_full.jl")

using Base.Threads

gr(grid=false, ms=3)

### BUILD GRAPH
function buildgraph(R, f1, f3; tol=1e-2; B=0) 
    f2 = 1.0-f1-f3
    α = 1-R
    k = floor(Int, (2-f1+f3)/α)
    s = k+1-(2-f1+f3)/α
    K = [fill(0,k-1); s; 1-s]
    Λ = [f1, f2, f3]
    m, n, nedges, rho, lambda = valid_degrees(K,Λ,3*5*7, B=1)
    if B!=0
        B = 10^max(0, 3-round(Int,log10(n)))
    end
    m, n, nedges, rho, lambda = valid_degrees(K,Λ,3*5*7, B=B)
    @assert isapprox(Λ, lambda, atol=tol)
    @assert isapprox(lambda[1], f1, atol=tol)
    @assert isapprox(R, rate(lambda, rho), atol=tol)
    m, n, nedges, rho, lambda
end

R = 0.5
f3 = 0.2
f1s = [0, 0.01, 0.02, 0.03]

for f1 in f1s
    m, n, nedges, rho, lambda = buildgraph(R, f1, f3)
    @show f1, n
end

### BP + dec
function bp_dec(n, m, nedges, rho, lambda, β; Tmax=1, fair_decimation=true,
        verbose=true)
    s = rand((-1,1), n)
    efield = [(exp(β*ss),exp(-β*ss)) for ss in s]
    bp = bp_full(n, m, nedges, lambda, rho, efield)
    B, indep = findbasis_slow(Array(bp.H))
    R = size(B,2)/n
    dist = decimate!(bp, efield, indep, s, B, maxiter=10^3, Tmax=Tmax, tol=1e-5, 
        fair_decimation=fair_decimation, verbose=verbose) 
    dist, bp, R
end

### COMPUTE
β = 3.0
navg = 10
Tmax = 20
D_bp = [zeros(navg) for _ in f1s]
R_bp = zeros(length(f1s))

for i in eachindex(f1s)
    println("Degree profile $i of ", length(f1s))
    m, n, nedges, rho, lambda = buildgraph(R, f1s[i], f3)
    @threads for j in 1:navg
        D_bp[i][j], _, R_bp[i] = bp_dec(n, m, nedges, rho, lambda, β; Tmax=Tmax)
    end 
end

pyplot();
pl = plot_rdb();
for i in eachindex(f1s)
    lab = "BP+dec - f1=$(f1s[i])"
    scatter!(pl, repeat([R_bp[i]], navg), D_bp[i], label=lab, ms=3, c=:purple)
end
pl2 = plot_rdb();
for i in eachindex(f1s)
    lab = "BP+dec - f1=$(f1s[i])"
    scatter!(pl, [R_bp[i]], [mean(D_bp[i])], label=lab, ms=3, c=:purple)
end
pl3 = plot(pl, pl2, size=(900,300), margin=5Plots.mm);
savefig(pl3, "bp_dec_leaves.pdf")
