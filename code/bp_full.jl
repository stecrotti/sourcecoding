include("slim_graphs.jl")
include("cavity.jl")
include("bp.jl")

struct BPFull{F,M}
    H :: SparseMatrixCSC{F,Int}     # size (nfactors,nvars)
    X :: SparseMatrixCSC{Int,Int}   # to get neighbors of factor nodes
    h :: Vector{M}                  # messages var -> factor_perm
    u :: Vector{M}                  # messages factor -> var
    efield :: Vector{M}             # external field
    belief :: Vector{M}
end
nfactors(bp::BPFull) = size(bp.H,1)
nvars(bp::BPFull) = size(bp.H,2)
function BPFull(H::SparseMatrixCSC)
    n = size(H,2)
    X = sparse(SparseMatrixCSC(size(H)...,H.colptr,H.rowval,collect(1:length(H.nzval)))')
    efield = fill((0.5,0.5), n)
    h = fill((0.5,.5),nedges)
    u = fill((0.5,.5),nedges)
    belief = fill((0.5,.5),n)
    BPFull(H, X, h, u, efield, belief)
end

function bp_full(n, m, nedges, Lambda, Rho, efield=fill((0.5,0.5),n), 
        h=fill((0.5,.5),nedges), u=fill((0.5,.5),nedges),  
        belief=fill((0.5,.5),n), args...; kw...)
    H = sparse(ldpc_matrix(n, m, nedges, Lambda, Rho, args...; kw...)')
    X = sparse(SparseMatrixCSC(size(H)...,H.colptr,H.rowval,collect(1:length(H.nzval)))')
    BPFull(H, X, h, u, efield, belief)
end

msg_conv(h1::Tuple, h2::Tuple) = (h1[1]*h2[1]+h1[2]*h2[2], h1[1]*h2[2]+h1[2]*h2[1]) 
msg_mult(u1::Tuple, u2::Tuple) = u1 .* u2

function update_var!(bp::BPFull, i::Int; damp=0.0, rein=0.0)
    ε = 0.0
    ∂i = nzrange(bp.H, i)
    u = bp.u[∂i]
    h = copy(bp.h[∂i])
    full = cavity!(h, u, msg_mult, bp.efield[i])
    iszero(sum(full)) && return -1  # normaliz of belief is zero
    bp.belief[i] = full ./ sum(full)
    for (hnew,a) in zip(h, ∂i)
        hnew = hnew ./ sum(hnew)
        ε = max(ε, maximum(abs, hnew.-bp.h[a]))
        bp.h[a] = bp.h[a].*damp .+ hnew.*(1-damp)
    end
    bp.efield[i] = bp.efield[i] .+ rein.*bp.belief[i]
    ε
end

function update_factor!(bp::BPFull, a::Int; damp=0.0)
    ε = 0.0
    ∂a = nonzeros(bp.X)[nzrange(bp.X, a)]
    u = copy(bp.u[∂a])
    h = bp.h[∂a]
    full = cavity!(u, h, msg_conv, (1.0,0.0))
    for (unew,i) in zip(u, ∂a)
        unew = unew ./ sum(unew)    # should not be needed
        ε = max(ε, maximum(abs, unew.-bp.u[i]))
        bp.u[i] = bp.u[i].*damp .+ unew.*(1-damp)
    end
    ε
end

function iteration!(bp::BPFull; maxiter=10^3, tol=1e-12, damp=0.0, rein=0.0,
        callback=(x...)->false)
    errf = fill(0.0, size(bp.H,1))
    errv = fill(0.0, size(bp.H,2))
    ε = 0.0
    for it = 1:maxiter
        for a=1:size(bp.H,1)
            errf[a] = update_factor!(bp, a, damp=damp)
            errf[a] == -1 && return -1,it
        end
        for i=1:size(bp.H,2)
            errv[i] = update_var!(bp, i, damp=damp, rein=rein)
            errv[i] == -1 && return -1,it
        end
        ε = max(maximum(errf), maximum(errv))
        callback(it, ε, bp) && return ε,it
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


function parity(bp::BPFull, x::AbstractVector)
    z = sum(bp.H*x .% 2)
    return z 
end
function distortion(x::AbstractVector, y::AbstractVector)
    d = 0
    for (xx,yy) in zip(x,y)
        d += sign(xx)!=sign(yy)
    end
    d/length(x)
end
function performance(bp::BPFull, s::AbstractVector)
    x = argmax.(bp.belief) .== 2
    nunsat = parity(bp, x)
    y = s .== -1
    dist = mean(x .!= y)
    ovl = 1-2*dist
    nunsat, ovl, dist
end

#### DECIMATION

# try Tmax times to reach zero unsat with decimation
# returns nunsat, ovl, dist
function decimate!(bp::BPFull, efield, indep, s; Tmax=1, kw...)
    freevars = falses(nvars(bp)); freevars[indep] .= true
    for t in 1:Tmax
        ε, nunsat, ovl, dist, iters = decimate1!(bp, efield, freevars, s; kw...)
        print("Trial $t of $Tmax: ")
        ε == -1 && print("contradiction found. ")
        println(nunsat, " unsat. Dist = ", round(dist,digits=3))
        nunsat == 0 && return nunsat, ovl, dist
        freevars .= false; freevars[indep] .= true
    end
    return -1, NaN, NaN
end

# 1 trial of decimation
function decimate1!(bp::BPFull, efield, freevars::BitArray{1}, s; 
        u_init=fill((0.5,0.5), length(bp.u)), h_init=fill((0.5,0.5), length(bp.h)),
        callback=(ε,nunsat,args...) -> (ε==-1||nunsat==0), kw...)
    # reset messages
    bp.h .= h_init; bp.u .= u_init
    bp.efield .= efield
    # warmup bp run
    ε, iters = iteration!(bp; kw...)
    nunsat, ovl, dist = performance(bp, s)
    nfree = sum(freevars)
    callback(ε, nunsat, bp, nfree, ovl, dist, iters, 0, -Inf) && return ε, nunsat, ovl, dist, iters

    for t in 1:nfree
        maxfield, tofix, newfield = find_most_biased(bp, freevars)
        freevars[tofix] = false
        # fix most decided variable by applying a strong field 
        bp.efield[tofix] = newfield
        ε, iters = iteration!(bp; kw...)
        nunsat, ovl, dist = performance(bp, s)
        callback(ε, nunsat, bp, nfree-t, ovl, dist, iters, t, maxfield) && return ε, nunsat, ovl, dist, iters
    end
    ε, nunsat, ovl, dist, iters
end

# returns max prob, idx of variable, sign of the belief
function find_most_biased(bp::BPFull, freevars::BitArray{1})
    m = -Inf; mi = 1; s = 0
    newfields = [(1.0,0.0), (0.0,1.0)]
    newfield = (0.5,0.5)
    for (i,h) in pairs(bp.belief)
       if freevars[i] && maximum(h)>m
            m, q = findmax(h); mi = i
            newfield = newfields[q]
       end
    end
    m, mi, newfield
end

function cb_decimation(ε, nunsat, bp::BPFull, nfree, ovl, dist, iters, step, maxfield, args...)
    @printf(" Step  %3d. Free = %3d. Maxfield = %1.2E. ε = %6.2E. Unsat = %3d. Ovl = %.3f. Iters %d\n", 
            step, nfree, maxfield, ε, nunsat,  ovl, iters)
    return ε==-1 || nunsat==0
end


