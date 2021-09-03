using SparseArrays, Random, Printf, Plots

include("matrix_generator.jl")
include("slim_graphs.jl")   # methods to compute basis

struct BeliefPropagation{F,M}
    H :: SparseMatrixCSC{F,Int}     # size (nvars,nfactors)
    m :: Vector{M}                  # messages (parametrized with magnetization)
    efield :: Vector{M}             # external field (parametrized with magnetization)
end
nfactors(bp::BeliefPropagation) = size(bp.H,2)
nvars(bp::BeliefPropagation) = size(bp.H,1)


function belief_propagation(n, m, nedges, Lambda, Rho, efield=zeros(n), 
        msg=zeros(nedges),  args...; kw...)
    H = ldpc_matrix(n, m, nedges, Lambda, Rho, args...; kw...)
    BeliefPropagation(H, msg, copy(efield))
end

# BP ROUTINES
struct Prod{T}
    p::T
    n::Int  # number of zeros
end
Prod{T}() where T = Prod(one(T), 0)
Base.:*(P::Prod{T}, x) where T = iszero(x) ? Prod(P.p, P.n+1) : Prod(P.p * x, P.n)
function Base.:/(P::Prod{T}, x) where T
    if iszero(x)
        return P.n==1 ? P.p : zero(T)
    else
        return P.n==0 ? P.p/x : zero(T)
    end
end

# returns -1 if a contradiction is found, the max absolute change in message otherwise
function update_factor!(bp::BeliefPropagation, a::Int; damp=0.0, rein=0.0)
    maxchange = 0.0
    t = Prod{Float64}()
    vars = rowvals(bp.H)
    for i in nzrange(bp.H,a)
        v = vars[i]
        # if bp.efield[v] == bp.m[i]
        #     # avoid 0/0 when the two are equal and have absolute value 1
        #     bp.efield[v] = 0.0
        if abs(bp.efield[v])!=1
        # else
            bp.efield[v] = (bp.efield[v]-bp.m[i])/(1-bp.efield[v]*bp.m[i])
        end
        t *= bp.efield[v]
    end
    for i in nzrange(bp.H,a)
        v = vars[i]
        m = t/bp.efield[v]
        maxchange = max(maxchange, abs(m-bp.m[i]))
        m = m*(1-damp) + bp.m[i]*damp
        newfield = (m+bp.efield[v])/(1+m*bp.efield[v])
        # contradiction: m and field[v] were completely polarized but opposite
        isnan(newfield) && return -1.0
        bp.m[i] = m
        bp.efield[v] = sign(newfield)*abs(newfield)^(max(1.0-rein,0.0))
    end
    maxchange
end

# max-sum
function update_factor_ms!(bp::BeliefPropagation, a::Int; damp=0.0, rein=0.0)
    maxchange = 0.0
    fmin = fmin2 = Inf
    imin = 1
    s = Prod{Int}()
    vars = rowvals(bp.H)
    for i in nzrange(bp.H,a)
        v = vars[i]
        if bp.efield[v] == bp.m[i]
            # avoid 0/0 when the two are equal and have absolute value Inf
            bp.efield[v] = 0
        else
            bp.efield[v] = bp.efield[v] - bp.m[i]
        end
        s *= sign(bp.efield[v])
        m = abs(bp.efield[v])
        if fmin > m
            fmin2 = fmin
            fmin = m
            imin = i
        elseif fmin2 > m
            fmin2 = m
        end
    end
    for i in nzrange(bp.H,a)
        v = vars[i]
        m = (i == imin ? fmin2 : fmin) * (s / sign(bp.efield[v]))
        maxchange = max(maxchange, abs(m-bp.m[i]))
        m = m*(1-damp) + bp.m[i]*damp
        newfield = m + bp.efield[v]
        # contradiction: m and field[v] were completely polarized but opposite
        isnan(newfield) && return -1.0
        bp.m[i] = m
        bp.efield[v] = newfield*(1.0+rein)
    end
    maxchange
end

# returns ε=-1 if a contradiction is found and the number of the last iteration
function iteration!(bp::BeliefPropagation; factor_perm=randperm(nfactors(bp)), 
        maxiter=1000, tol=1e-12, damp=0.0, rein=0.0, callback=(x...)->false,
        update! = update_factor!)
    ε  = 0.0
    for it in 1:maxiter 
        ε  = 0.0
        for a in factor_perm
            maxchange = update!(bp, a, damp=damp, rein=rein*it)
            maxchange == -1 && return -1.0, it
            ε = max(ε, maxchange)
        end
        shuffle!(factor_perm)
        callback(it, ε, bp) && return ε, it
        ε < tol && return ε, it
    end
    ε, maxiter
end

# given a basis and the values of x[indep], compute what x[dep] must be
function fix_indep!(x, B, indep)
    n,k = size(B)
    dep = setdiff(1:n, indep)
    x[dep] .= B[dep,:]*x[indep] .% 2
    σ = 1 .- 2x
end

function parity(bp::BeliefPropagation, x::AbstractVector)
    z = p = 0
    vars = rowvals(bp.H)
    for a in 1:nfactors(bp)
        for i in nzrange(bp.H,a)
            p += x[vars[i]]
        end
        z += p % 2  
        p = 0
    end
    return z 
end
function distortion(x::AbstractVector, y::AbstractVector)
    d = 0.0
    for (xx,yy) in zip(x,y)
        d += sign(xx)!=sign(yy)
    end
    d/length(x)
end
function performance(bp::BeliefPropagation, fields, 
        x = sign.(bp.efield) .== -1)
    x .= sign.(bp.efield) .== -1
    nunsat = parity(bp, x)
    dist = distortion(fields, bp.efield)
    ovl = 1-2*dist
    nunsat, ovl, dist
end


#### DECIMATION

# try Tmax times to reach zero unsat with decimation
function decimate!(bp::BeliefPropagation, fields, indep; Tmax=1, kw...)
    freevars = falses(nvars(bp)); freevars[indep] .= true
    for t in 1:Tmax
        print("Trial $t of $Tmax: ")
        ε, nunsat, ovl, dist, iters = decimate1!(bp, fields, freevars; kw...)
        ε == -1 && print("contradiction found. ")
        println(nunsat, " unsat")
        nunsat == 0 && return nunsat, ovl, dist
        freevars .= false; freevars[indep] .= true
    end
    return -1, NaN, NaN
end

# 1 trial of decimation
function decimate1!(bp::BeliefPropagation, fields, freevars::BitArray{1}; 
        callback=(ε,nunsat,args...) -> (ε==-1||nunsat==0), kw...)
    # reset messages
    bp.m .= 0; bp.efield .= copy(fields)
    # pre-allocate for speed
    factor_perm = randperm(nfactors(bp)); x=falses(nvars(bp))
    # warmup bp run
    ε, iters = iteration!(bp, factor_perm=factor_perm; kw...)
    nunsat, ovl, dist = performance(bp, fields, x)
    nfree = sum(freevars)
    callback(ε, nunsat, bp, nfree, ovl, dist, iters, 0) && return nunsat, ovl, dist, iters

    for t in 1:nfree
        maxfield, tofix = find_most_biased(bp, freevars)
        freevars[tofix] = false
        # if tofix is undecided, give it its value in the source    
        bp.efield[tofix] = maxfield==0 ? fields[tofix] : sign(bp.efield[tofix])
        ε, iters = iteration!(bp, factor_perm=factor_perm; kw...)
        nunsat, ovl, dist = performance(bp, fields  , x)
        callback(ε, nunsat, bp, nfree-t, ovl, dist, iters, t) && return ε, nunsat, ovl, dist, iters
    end
    ε, nunsat, ovl, dist, iters
end

function find_most_biased(bp::BeliefPropagation, freevars::BitArray{1})
    m = -Inf; mi = 1
    for (i,h) in pairs(bp.efield)
       if freevars[i] && abs(h)>m
            m = abs(h); mi = i
       end
    end
    m, mi
end

function cb_decimation(ε, nunsat, bp::BeliefPropagation, nfree, ovl, dist, iters, step)
    @printf(" Step  %3d. Free = %3d. ε = %6.2E. Unsat = %3d. Ovl = %.3f. Iters %d\n", 
            step, nfree, ε, nunsat,  ovl, iters)
    (ε==-1 || nunsat==0) && return true
    false
end



