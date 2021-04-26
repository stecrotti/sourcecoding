include("FactorGraph.jl")
include("ldpc_graph.jl")

using ProgressMeter

mutable struct Seaweed{T}
    fg :: FactorGraphGF2
    toflip :: T
    Qf :: Vector{Tuple{Int,Int}}
    Qv_up :: Vector{Tuple{Int,Int}}
    Qv_down :: Vector{Tuple{Int,Int}}
    depths :: Vector{Int}
    isincore :: T
end

function Seaweed(fg::FactorGraphGF2)
    n = fg.n
    toflip = spzeros(Bool, n)
    Qf = Tuple{Int,Int}[]; Qv_up = Tuple{Int,Int}[]; Qv_down = Tuple{Int,Int}[]
    depths = lr(fg)[1]
    isincore = sparse(depths .== 0)
    nnz(isincore) > 0 && @warn("Non-empty core")
    Seaweed(fg, toflip, Qf, Qv_up, Qv_down, depths, isincore)
end

function isbelow(e::Int, v::Int, fg::FactorGraph, depths::Vector{Int})
    depths[v] == minimum(depths[fg.Fneigs[e]])
end

function grow_sw!(sw::Seaweed, seed::Int; 
        callback=sw->nothing)
    # Check that there is at least 1 leaf
    @assert nvarleaves(sw.fg) > 0 "Graph must contain at least 1 leaf"
    # Check that seed is one of the variables in the graph
    @assert seed <= sw.fg.n "Cannot use var $seed as seed since FG has $(fg.n) variables"
    # Check that seed is a variable outside the core    
    @assert !sw.isincore[seed] "Cannot grow seaweed starting from a variable in the core"
    # Refresh seaweed to all zeros
    sw.toflip .= false
    push!(sw.Qf, (0, seed))
    while !(isempty(sw.Qf))
        f,v = pop!(sw.Qf); growf!(sw, f, v)
        if !isempty(sw.Qv_up) 
            f,v = pop!(sw.Qv_up)
            growv_up!(sw, f, v)
        end
        if !isempty(sw.Qv_down)
            f,v = pop!(sw.Qv_down)
            growv_down!(sw, f, v)
        end
        callback(sw)
    end
    # check that the resulting seaweed satisfies parity
    @assert parity(sw.fg, sw.toflip)==0
    return dropzeros!(sw.toflip)
end
cb_seaweed = sw->println("Weight ", sum(sw.toflip))

function growf!(sw::Seaweed, f::Int, v::Int)
    sw.toflip[v] = !sw.toflip[v]
    neigs_of_v = sw.fg.Vneigs[v]
    # find the (at most one) factor below v
    i = findfirst(ee->isbelow(ee, v, sw.fg, sw.depths) && ee!=f, neigs_of_v)
    # grow upwards branches everywhere except where you came from (factor f) and
    #  factor below (factor with index i)
    for (j,ee) in enumerate(neigs_of_v); if j!=i && ee!=f
        push!(sw.Qv_up, (ee, v))
    end; end
    # grow downwards to maximum 1 factor
    if !isnothing(i)
        push!(sw.Qv_down, (neigs_of_v[i], v))
    end
    nothing
end

function growv_up!(sw::Seaweed, f::Int, v::Int)
    mindepth = typemax(Int)
    neigs_of_f = sw.fg.Fneigs[f]
    for w in neigs_of_f
        if w != v && !sw.isincore[w] && sw.depths[w]<mindepth
            mindepth = sw.depths[w]
        end
    end
    argmindepth = [w for w in neigs_of_f 
        if w != v && !sw.isincore[w] && sw.depths[w]==mindepth]
    if !isempty(argmindepth) 
        new_v = rand(argmindepth)
        push!(sw.Qf, (f, new_v))
    end
    nothing
end

function growv_down!(sw::Seaweed, f::Int, v::Int)
    # consider all neighbors except the one we're coming from (v)
    neigs_of_f = [w for w in sw.fg.Fneigs[f] if (w != v && !sw.isincore[w])]
    if !isempty(neigs_of_f)
        # if v is not the only neighbor of minimum depth, pick one of the others
        mindepth = minimum(sw.depths[neigs_of_f])
        if sw.depths[v]==mindepth
            argmindepth = findall(sw.depths[neigs_of_f] .== mindepth)
            # pick at random one of the neighbors with equal min depth
            new_v = neigs_of_f[rand(argmindepth)]
            push!(sw.Qf, (f, new_v))
        else
            # find maximum depth
            maxdepth = maximum(sw.depths[neigs_of_f])
            argmaxdepth = findall(sw.depths[neigs_of_f] .== maxdepth)
            # pick at random one of the neighbors with equal max depth
            new_v = neigs_of_f[rand(argmaxdepth)]
            push!(sw.Qf, (f, new_v))
        end
    else
        println("Warning: no neighbors found going down from v to f")
    end
    nothing
end

# counts the number of L.I. columns of A
function nindepcols!(A)
    gfrcefGF2!(A)
    cnt = 0
    for j in 1:size(A,2)
        cnt += !iszero(A[:,j])
    end
    cnt
end

function seaweed_basis(nsw, fg, dim, Baux = sparse(falses(fg.n,dim)))
    B = sparse(falses(fg.n, dim))
    cnt = 1
    v = sparse(falses(fg.n))
    # instanciate seaweed, depths and everything
    sw = Seaweed(fg)
    seed_perm = findall(.!sw.isincore)
    prog = Progress(nsw)
    for i in 1:nsw
        seed = seed_perm[mod1(i,fg.n)]
        # grow seaweed and put it as the next free column in Baux
        v .= grow_sw!(sw,seed)
        Baux[:,cnt] .= v
        nind = nindepcols!(Baux)
        if nind==cnt
            # if seaweed not in span, add it
            B[:,cnt] .= v
            # update Baug, ready for a new gaussian elim
            B[:,1:cnt] .= B[:,1:cnt]
            cnt += 1
        end
        # if the whole space is spanned, return
        if cnt == dim+1
            return B, i
        end
        i==fg.n && shuffle!(seed_perm)
        sw.toflip .= false
        empty!(sw.Qv_up); empty!(sw.Qv_down)
        ProgressMeter.next!(prog; showvalues = [(:iter,i), (:dim, cnt-1)])
    end
    @warn "Not enough iterations to span the whole space"
    return dropzeros!(B), nsw
end
