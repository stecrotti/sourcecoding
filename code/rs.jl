using OffsetArrays

# MESSAGE UPDATES

function iter_factor(q, k)
    p = zero(q)
    N = lastindex(q)
    for f = 1:N
        v1 = 2q[f]
        v2 = 2*sum(q[f+1:N])
        v = (v1+v2)^k-v2^k
        p[+f] = v/2
        p[-f] = v/2
    end
    p[0] = 1-(1-q[0])^k
    p ./ sum(p)
end

# Convolution with clamping
function convolve(p1, p2)
    N = max(lastindex(p1), lastindex(p2))
    q = fill(0.0,-N:N)
    for f1 ∈ eachindex(p1)
        for f2 ∈ eachindex(p2)
            q[clamp(f1+f2, -N, N)] += p1[f1]*p2[f2]
        end
    end
    q ./= sum(q)
end

function iter_var(p,d)
    q = zero(p)
    q[-1:1] += [0.5, 0, 0.5]
    for d1 ∈ 1:d
        q = convolve(q, p)
    end
    q
end


# OVERLAP

function overlap_factor(q,k) 
    2^k * sum(h*(sum(q[h:end])^k-sum(q[h+1:end])^k) for h=1:lastindex(q))
end

# RS COMPUTATION

function RS(Pk, Λ; N=100, tol=1e-5, maxiter=100, damp=0.9)
    ks = [k for k in eachindex(Pk) if Pk[k] > tol]
    ds = [d for d in eachindex(Λ) if Λ[d] > tol]
    @assert sum(Pk[ks]) ≈ 1 && sum(Λ[ds]) ≈ 1
    p = fill(1.0, -N:N); p ./= sum(p)
    for iter=1:maxiter
        q = sum(d*Λ[d]*iter_var(p, d-1) for d=ds)
        q ./= sum(q)
        p1 = sum(k*Pk[k]*iter_factor(q, k-1) for k=ks)
        p1 ./= sum(p1)
        err = maximum(abs, p1 - p); err < tol && (@show err iter; break)
        p .= p .* damp .+ p1 .* (1-damp)
    end
    α = sum(d*Λ[d] for d=ds) / sum(k*Pk[k] for k=ks)
    
    q = sum(d*Λ[d]*iter_var(p, d-1) for d=ds); #fia
    O = sum(abs(f)*q[f] for f=eachindex(q))
    q2 = sum((d-1)*Λ[d]*iter_var(p, d) for d=ds); #fi
    O -= sum(abs(f)*q2[f] for f=eachindex(q2))
    q ./= sum(q)
    O -= α*sum(Pk[k]*overlap_factor(q,k) for k=ks) #fa
    1-α, (1-O)/2, p  
end


# SLOW VERSIONS

function iter_slow_factor(q, k)
    N = lastindex(q)
    p = fill(0.0, -N:N)
    for fs ∈ Iterators.product(fill(-N:N,k)...)
        f = minimum(abs.(fs))*sign(prod(fs))
        p[clamp(f, -N, N)] += prod(q[f1] for f1 ∈ fs)
    end
    p ./= sum(p)
end

function iter_slow_var(p, d) 
    N = lastindex(p)
    q = fill(0.0, -N:N)
    for fs in Iterators.product(fill(-N:N,d)...)
        f = sum(fs)
        prob = 1/2*prod(p[f] for f ∈ fs)
        for s ∈ (-1,1)
            q[clamp(f + s, -N, N)] += prob
        end
    end
    q ./= sum(q)
end