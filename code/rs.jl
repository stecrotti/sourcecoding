using OffsetArrays

# MESSAGE UPDATES

function iter_factor(q, k, N = lastindex(q))
    p = fill(zero(eltype(q)), -N:N)
    for f = 1:N
        v1 = 2q[f]
        v2 = 2*sum(q[f+1:end])
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
    q = fill(zero(promote_type(eltype(p1),eltype(p2))),-N-1:N+1)
    for f1 ∈ eachindex(p1)
        for f2 ∈ eachindex(p2)
            q[clamp(f1+f2, -N-1, N+1)] += p1[f1]*p2[f2]
        end
    end
    q
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
    sum(h*((sum(q[h:end]))^k-(sum(q[h+1:end]))^k) for h=1:lastindex(q))
end

# RS COMPUTATION

function RS(Pk, Λ; N=100, tol=1e-5, maxiter=100, damp=0.9, T=Float64, 
        verbose=true, Fs=zeros(T, 3), p = fill(one(T), -N:N))
    ks = [k for k in eachindex(Pk) if Pk[k] > tol]
    ds = [d for d in eachindex(Λ) if Λ[d] > tol]
    # @assert sum(Pk[ks]) ≈ 1 && sum(Λ[ds]) ≈ 1
    p = copy(p)
    p ./= sum(p)
    q = copy(p)
    for iter=1:maxiter
        q1 = copy(q)
        q = sum(d*Λ[d]*iter_var(p, d-1) for d=ds)
        q ./= sum(q)
        p1 = sum(k*Pk[k]*iter_factor(q, k-1, N) for k=ks)
        p1 ./= sum(p1)
        # err = max(maximum(abs, q1 - q), maximum(abs, p1 - p))
        err = maximum(abs, p1 - p)
        # err < tol && (verbose && @show err iter; break)
        p .= p .* damp .+ p1 .* (1-damp)
    end
    α = sum(d*Λ[d] for d=ds) / sum(k*Pk[k] for k=ks)
    
    q = sum(d*Λ[d]*iter_var(p, d-1) for d=ds); #fia
    Fia = -sum(abs(f)*q[f] for f=eachindex(q)); verbose && @show Fia
    O = sum(abs(f)*q[f] for f=eachindex(q))
    q2 = sum((d-1)*Λ[d]*iter_var(p, d) for d=ds); #fi
    Fi = sum(abs(f)*q2[f] for f=eachindex(q2)); verbose && @show Fi
    O -= sum(abs(f)*q2[f] for f=eachindex(q2))
    q ./= sum(q)
    safe2 = convert(T, 2)
    O -= α*sum(Pk[k]*safe2^k*overlap_factor(q,k) for k=ks) #fa
    Fa = α*sum(Pk[k]*safe2^k*overlap_factor(q,k) for k=ks); verbose && @show Fa
    verbose && @show O

    Fs .= [Fa, Fi, Fia]
    1-α, (1-O)/2, p, q
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


f_p1(k, p1) = 1/2*((1+(1-p1)^k-3p1)^k-((1-p1)^k-p1)^k)

### WITH J=2
function P0_P1(k; p1_0=2/k, damp=0.9, N=5*10^4, tol=1e-20)
   p1 = p1_0
   for it in 1:N
        p1_new = f_p1(k, p1)
        err = abs(p1 - p1_new)
        it == N && @show err
        err < tol && break
        p1 = damp*p1 + (1-damp)*p1_new
        
   end
   p0 = 1-(1-p1)^k
   p0, p1
end

function RS_quick(k::Integer, J::Val{2}; kw...)
    p0, p1 = P0_P1(k-1; kw...)
    p2 = (1 - p0 - 2p1) / 2
    @assert p2 ≥ 0
    p = OffsetArray([p2, p1, p0, p1, p2], -2:2)
    q0 = p1; q1 = (1-2p1)/2; q2 = p1/2; q3 = (1-p0-2p1)/2
    q = OffsetArray([q3, q2, q1, q0, q1, q2, q3], -3:3)
    p, q
end

f_p0(k, p0) = 1 - ((1+p0)/2)^k

function P0(k; p0_0=0.5, damp=0.9, N=10^4, tol=1e-16)
    t = 1 - p0_0
    for _ in 1:N
        tnew = (1-t/2)^k
        abs(t - tnew) < tol && break
        t = damp*t + (1-damp)*tnew
    end
    1-t
end

function RS_quick(k::Integer, J::Val{1}; kw...)
    p0 = P0(k-1; kw...)
    p1 = (1 - p0)/2
    p = OffsetArray([p1, p0, p1], -1:1)
    q0 = p1; q1 = p0/2; q2 = p1/2
    q = OffsetArray([q2, q1, q0, q1, q2], -2:2)
    p, q
end