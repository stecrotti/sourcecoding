include("../bp_full.jl")

using Base.Threads

gr(grid=false, ms=3)

### BUILD GRAPH
function buildgraph(R, f1, f3; tol=1e-2) 
    f2 = 1.0-f1-f3
    α = 1-R
    k = floor(Int, (2-f1+f3)/α)
    s = k+1-(2-f1+f3)/α
    K = [fill(0,k-1); s; 1-s]
    Λ = [f1, f2, f3]
    m, n, nedges, rho, lambda = valid_degrees(K,Λ,3*5*7, B=1)
    B = max(0, 3-round(Int,log10(n)))
    m, n, nedges, rho, lambda = valid_degrees(K,Λ,3*5*7, B=10^B)
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

function ms_dec(n, m, nedges, K, Λ; verbose=true, maxiter=10^2, Tmax=1, kw...)
    efield = [(.0,.0) for _ in 1:n]
    s = zeros(Int, n)
    vars = rand(1:n, n*2÷3); factors=rand(1:m-1, m*2÷3)
    Ht = ldpc_matrix(n, m, nedges, Λ, K; accept_multi_edges=false)[:,1:end-1]
    H = permutedims(Ht)
    s .= rand((-1.,1.),n)
    d = Inf
    for t in 1:Tmax
        efield .= [(si,-si).+ 1e-5.*(randn(),randn()) for si in s]
        ms = BPFull(H, efield)
        ε, iters = iteration_ms!(ms, maxiter=maxiter;
            vars=vars, factors=factors, kw...)
        nunsat, ovl, dist = performance(ms, s)
        verbose && @show nunsat, ovl, dist
        flush(stdout)
        if nunsat!=0
            B,indep = findbasis(H, Ht)
            x = argmax.(ms.belief) .== 2
            σ = fix_indep!(x, B, indep)   
            d = min(d, distortion(σ,s))
        else
            d = min(d, dist)
        end
    end
    d
end

### COMPUTE
navg = 10
Tmax = 1
D_ms = [zeros(navg) for _ in f1s]
R_ms = zeros(length(f1s))

for i in eachindex(f1s)
    println("Degree profile $i of ", length(f1s))
    m, n, nedges, rho, lambda = buildgraph(R, f1s[i], f3)
    R_ms[i] = 1 - m/n
    @threads for j in 1:navg
        D_ms[i][j] = ms_dec(n, m, nedges, rho, lambda; Tmax=Tmax, maxiter=5*10^3, 
            rein=1e-3, verbose=false)
    end 
end

pl = plot_rdb();
for i in eachindex(f1s)
    lab = "MS+rein - f1=$(f1s[i])"
    scatter!(pl, repeat([R_ms[i]], navg), D_ms[i], label=lab, ms=3, c=:purple)
end
pl2 = plot_rdb();
for i in eachindex(f1s)
    lab = "MS+rein - f1=$(f1s[i])"
    scatter!(pl, [R_ms[i]], [mean(D_ms[i])], label=lab, ms=3, c=:purple)
end
# pl3 = plot(pl, pl2, size=(900,300), margin=5Plots.mm);
# savefig(pl3, "ms_rein_leaves.pdf");