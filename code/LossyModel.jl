#### A wrapper for a FactorGraph object with temperature(s), current guess for the solution, source vector ####
using LinearAlgebra, Lazy

mutable struct LossyModel
    fg::FactorGraph     # Factor graph instance
    x::Vector{Int}      # Current state
    beta1::Real         # Inverse temperature for checks
    beta2::Real         # Inverse temperature for overlap with input vector y
    y::Vector{Int}      # Vector to be compressed
end

# 'Inherit' methods from inner property `fg`
@forward LossyModel.fg adjmat, nullspace, rank, isfullrank, newbasis, 
    permute_to_triangular

# Constructor for lossy model with LDPC matrix
function LossyModel(q::Int, n::Int, m::Int; beta1::Real=Inf, beta2::Real=1.0,
        nedges::Int=generate_polyn(n,m)[1], lambda::Vector{T}=generate_polyn(n,m)[2],
        rho::Vector{T}=generate_polyn(n,m)[3],
        fields = [Fun(1e-3*randn(q)) for v in 1:n], verbose=false,
        arbitrary_mult = false,
        randseed::Int=0,
        y::Vector{Int}=rand(MersenneTwister(randseed), 0:q-1, n)) where {T<:AbstractFloat}

    !ispow2(q) && warning("The value of q you inserted (q=$q) is not a power of 2")
    fg = ldpc_graph(q,n,m, nedges, lambda, rho, fields, verbose=false,
        arbitrary_mult=arbitrary_mult, randseed=randseed)
    x = zeros(Int, n)
    return LossyModel(fg, x, beta1, beta2, y)
end

function LossyModel(fg::FactorGraph)
    x = zeros(Int, fg.n)
    beta1 = Inf
    beta2 = 1.0
    y = rand(0:fg.q-1, fg.n)
    return LossyModel(fg, x, beta1, beta2, y)
end

function Base.show(io::IO, lm::LossyModel)
    println(io, "Lossy compression model:")
    println(io, " - ", dispstring(lm.fg))
    println(io, " - Inverse temperatures β₁=$(lm.beta1) for checks and",
        " β₂=$(lm.beta2) for overlap")
end

function rate(lm::LossyModel) 
    n_indep_rows = rank(lm)
    r = 1 - n_indep_rows/lm.fg.n
    return r
end
function distortion(lm::LossyModel, x::Vector{Int}=lm.x)
    # return hd(x,lm.y)/(lm.fg.n*log2(lm.fg.q))
    return distortion(lm.fg, lm.y, x)
end
function log_nsolutions(lm::LossyModel, getbasis::Function=nullspace)::Int
    return size(getbasis(lm), 2)
end
nsolutions(lm::LossyModel, args...)::Int = lm.fg.q^log_nsolutions(lm, args...)

function breduction!(lm::LossyModel, args...; kwargs...)
    b = breduction!(lm.fg, args...; kwargs...)
    return b
end

# Support for general input x (can also be a matrix)
function paritycheck(lm::LossyModel, x::Array{Int,2}, varargin...)
    return paritycheck(lm.fg, x, varargin...)
end

# Input as a vector instead of 2d array
function paritycheck(lm::LossyModel, x::Vector{Int}=lm.x, varargin...)
    return paritycheck(lm.fg, x[:,:], varargin...)
end

function parity(lm::LossyModel, args...)
    return sum(paritycheck(lm, args...))
end

function energy(lm::LossyModel, x::Vector{Int}=lm.x)
    ener_checks = energy_checks(lm ,x)
    ener_overlap = energy_overlap(lm, x)
    return ener_checks + ener_overlap
end

function energy_checks(lm::LossyModel, x::Union{Vector{Int},Array{Int,2}}=lm.x;
                        f::Union{Int,Vector{Int},Array{Int,2}}=collect(1:lm.fg.m))
    # Unsatisfied checks
    z = paritycheck(lm, x, f)
    hw_checks = sum(hw(z))
    # In principle this should jsut be lm.beta1*hw_checks, but gotta take into
    #  account the case beta1=Inf, for which we choose the convention Inf*0=0
    ener_checks = hw_checks == 0 ? 0 : lm.beta1*hw_checks
end

function energy_overlap(lm::LossyModel, x::Union{Vector{Int},Array{Int,2}}=lm.x)
    return lm.beta2*hd(x, lm.y)
end

function refresh!(lm::LossyModel, args...; kwargs...)
    lm.x .= zeros(Int, lm.fg.n)
    return refresh!(lm.fg, lm.y, args...; kwargs...)
end

# Gaussian elimination on the graph
function gfrref!(lm::LossyModel)
    H = adjmat(lm)
    gfrref!(H, lm.fg.q, lm.fg.mult, lm.fg.gfdiv)
    lm.fg = FactorGraph(H)
    return nothing
end

function compress(lm::LossyModel, getbasis::Function=newbasis)
    indep = falses(lm.fg.n)
    # Return a basis plus store in `indep` the indices of indep variables
    nb = getbasis(lm, indep)
    x_compressed = lm.x[indep]
    return x_compressed
end

function decompress(x_compressed::Vector{Int}, basis::Vector{Int}, args...)
    x_reconstructed = gfmatrixmult(basis, x_compressed, args...)
    return x_reconstructed
end
function decompress(x_compressed::Vector{Int}, fg::FactorGraph, 
        getbasis::Function=newbasis)
    x_reconstructed = gfmatrixmult(getbasis(fg), x_compressed, q_mult_div(fg)[(1:2)]...)
end
function decompress(x_compressed::Vector{Int}, lm::LossyModel, 
        getbasis::Function=newbasis)
    return decompress(x_compressed, lm.fg, getbasis)
end


function weighted_full_adjmat(lm::LossyModel)
    @assert lm.fg.q == 2
    H = full_adjmat(lm.fg)
    for v in 1:lm.fg.n
        if lm.x[v] != lm.y[v]
            H[lm.fg.m+v,:] .*= -1
            H[:,lm.fg.m+v] .*= -1
        end      
    end
    return H
end

########
function lightweight_nullspace(lm::LossyModel; cutoff::Real=Inf, 
    verbose::Bool=false)
    # Start with a basis of the system
    oldbasis = nullspace(lm)
    hw_old = maximum([hw(collect(col)) for col in eachcol(oldbasis)])
    if verbose
        println("Finding a low-Hamming-weight basis...")
        println("\tThe basis I'm starting from has total Hamming weight ", hw_old,
        ".")
    end
    nsdim = size(oldbasis,2)
    allsolutions = enum_solutions(lm)
    newbasis = zeros(Int, size(oldbasis))
    g = solutions_graph(lm, cutoff=cutoff)
    if verbose
        conn = is_connected(g) ? "" : "NON "
         println("\tThe graph of solutions obtained with the",
            " required cutoff is ", conn, "CONNECTED.")
    end
    # Store all min bottleneck paths from 0 to the solutions in the old basis.
    # On graph g each node is a solution labelled with a number from 1 on.
    #  1 is the label for the zero vector.
    #  The index of the arrival node (the i-th basis vector) can be
    #  obtained recalling how the basis was constructed (linear combinations
    #  with all the possible strings of length n) => label(i)=lm.fg.q^(i-1)+1
    #  and keeping in mind that in Julia array indices start at 1.
    mbpaths = [min_bottleneck_path(g, 1, lm.fg.q^(i-1)+1) for i in 1:nsdim]
    mbpaths_idx = first.(mbpaths)
    mbpaths_val = last.(mbpaths)
    # Sort basis vectors so that small bottlenecks come first
    new_idx = sortperm(mbpaths_val)
    verbose && println("\tI'm sorting basis vector in order of ascending min ",
        "bottleneck. Values are: ", mbpaths_val[new_idx])
    sorted_mbpaths_idx = mbpaths_idx[new_idx]
    nadded = 0
    for i in 1:nsdim
        # vector containing the solutions in the path
        mbpath = allsolutions[sorted_mbpaths_idx[i]]
        # If there are other solutions on the path, try adding them to the basis
        if length(mbpath) > 0
            for t in 1:length(mbpath)-1
                # check if is not already in the span of newbasis
                hop = xor.(mbpath[t], mbpath[t+1])
                if !isinspan(newbasis, hop, q=lm.fg.q)
                    nadded += 1
                    newbasis[:,nadded] = hop
                    if nadded == nsdim 
                        hw_new = maximum([hw(collect(col)) for col in eachcol(newbasis)])
                        verbose && println("\tDone! The new basis has total ",
                        "Hamming weight ", hw_new)
                        return newbasis
                    end
                end
            end
        end
    end
    # If you got here, solutions were too far apart for this method to work
    verbose && println("\tCouldn't find a light basis, returning a generic one. ",
        "Try increasing the cutoff")
    return oldbasis
end

function min_bottleneck_path(g::AbstractGraph, source::Int, dest::Int)
    # get maximum ST
    mst_as_list = kruskal_mst(g, minimize=true)
    sources = [m.src for m in mst_as_list]
    dests = [m.dst for m in mst_as_list]
    weights = [m.weight for m in mst_as_list]
    mst_as_graph = SimpleWeightedGraph(sources, dests, weights)
    all_sps = dijkstra_shortest_paths(mst_as_graph, source)
    mbp = enumerate_paths(all_sps, dest)
    mbp_weights = [mst_as_graph.weights[mbp[i],mbp[i+1]] for i in 1:length(mbp)-1]
    mb = maximum(mbp_weights)
    return mbp, mb
end
