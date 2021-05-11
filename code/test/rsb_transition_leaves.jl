using OffsetArrays, Statistics
include("../bp_full.jl")
include("../bp.jl")

f1_0 = 0
f3 = 0.2
f2 = 1 - f1_0 -f3
Lambda = [f1_0,f2,f3]
K = [0,0,1]
n = 9000
nedges = Int(round(n*sum(i*l for (i,l) in pairs(Lambda)), digits=10))
m = Int(nedges/3);

# remove factors from matrix A until there is approx. a proportion f1 of leaves
function remove_factors(A, f1=1e-2)
    f = mean(isequal(1), sum(A,dims=1))
    while f < f1
        A = A[1:end-1,:]
        f = mean(isequal(1), sum(A,dims=1))
    end
    A
end

Hs = 1.1:0.02:2.1
navg = 10
f1 = 1e-2

err0 = fill(NaN,navg,length(Hs))
d0 = fill(NaN,navg,length(Hs))
err1 = fill(NaN,navg,length(Hs))
d1 = fill(NaN,navg,length(Hs)) 

for i in 1:navg
    println("### Instance $i of $navg")
    # extract instance
    s = rand((-1,1), n)
    A0 = sparse(ldpc_matrix(n, m, nedges, Lambda, K, accept_multi_edges=false)')
    bp0 = BPFull(A0)
    A1 = remove_factors(A0, f1)
    bp1 = BPFull(A1)
    for (j,H) in pairs(Hs)
        println("H=$H. $j of ", length(Hs))
        flush(stdout)
        bp0.efield .= [(exp(ss*H),exp(-ss*H)) for ss in s]
        fill!(bp0.h, (.5,.5)); fill!(bp0.u, (.5,.5)), fill!(bp0.belief, (.5,.5))
        err0[i,j], _ = iteration!(bp0, tol=1e-20, maxiter=5*10^4, damp=0.5, rein=0.0)
        d0[i,j] = avg_dist(bp0,s)

        bp1.efield .= [(exp(ss*H),exp(-ss*H)) for ss in s]
        fill!(bp1.h, (.5,.5)); fill!(bp1.u, (.5,.5)), fill!(bp1.belief, (.5,.5))
        err1[i,j], _ = iteration!(bp1, tol=1e-20, maxiter=5*10^4, damp=0.5, rein=0.0)
        d1[i,j] = avg_dist(bp1,s)
    end
end

err0_avg = vec(mean(err0,dims=1))
err0_sd = vec(std(err0,dims=1)/sqrt(navg))
err1_avg = vec(mean(err1,dims=1))
err1_sd = vec(std(err1,dims=1)/sqrt(navg))
d0_avg = vec(mean(d0,dims=1))
d0_sd = vec(std(d0,dims=1)/sqrt(navg))
d1_avg = vec(mean(d1,dims=1))
d1_sd = vec(std(d1,dims=1)/sqrt(navg));

plot(Hs, err0_avg, yerr=err0_sd, marker=:circle, label="f1=0", yscale=:log10);
plot!(Hs, err1_avg, yerr=err1_sd, marker=:circle, label="f1=0.01", yscale=:log10);
xlabel!("H"); ylabel!("err"); title!("Convergence error");
plot!(legend=:topleft, margin=15Plots.mm);
savefig("err")

plot(Hs, d0_avg, yerr=d0_sd, marker=:circle, label="f1=0");
plot!(Hs, d1_avg, yerr=d1_sd, marker=:circle, label="f1=0.01");
xlabel!("H"); ylabel!("Distortion"); title!("Expected distortion");
plot!(margin=15Plots.mm);
savefig("dist")

A0 = sparse(ldpc_matrix(n, m, nedges, Lambda, K, accept_multi_edges=false)')
A1 = remove_factors(A0, f1)

B0,_ = findbasis(A0[1:end-1,:])
R0 = size(B0,2)/n

B1,_ = findbasis(A1)
R1 = size(B1,2)/n

pp = plot_rdb();
nH = length(Hs)
scatter!(pp,fill(R0,nH),d0_avg, ms=3, label="f1=0");
scatter!(pp,fill(R1,nH),d1_avg, ms=3, label="f1=$f1");
# plot!(pp, legend=:outertopright)
pp2 = deepcopy(pp);
ylims!((.2,.3)); xlims!((.22,.32));
plot(pp2,pp, size=(800,350), margins=15*Plots.mm);
savefig("dist_rdb")


