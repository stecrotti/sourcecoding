using LightGraphs, SimpleWeightedGraphs, StatsBase, Plots, GraphRecipes

"""
    enum_solutions(H::Array{Int,2}, q::Int=2)

Computes all possible solutions of the homogenous linear system
Hy=0 on 𝔾𝔽(q)
"""
function enum_solutions(H::Array{Int,2}, q::Int=2)::Vector{Vector{Int}}
    !ispow(q, 2) && error("q must be a power of 2")
    !isgfq(H, q) && error("Matrix H has values outside GF(q) with q=$q")

    (m,n) = size(H)
    # Basis for the space of solutions
    ns = gfnullspace(H, q)
    k = size(ns,2)  # Nullspace dimension
    # All possible qᵏ coefficient combinations
    coeffs = allgfqstrings(q,k)
    # Multiply to get all possible linear combinations of basis vectors
    # Now solutions are stored as the columns of a matrix
    solutions_as_matrix = gfmatrixmult(ns, coeffs, q)
    # Convert to a vector of vectors
    return solutions_as_vector = [solutions_as_matrix[:,c] for c = 1:q^k]
end

enum_solutions(fg::FactorGraph) = enum_solutions(adjmat(fg), fg.q)
enum_solutions(lm::LossyModel) = enum_solutions(lm.fg)


"""
    allgfqstrings(q::Int, len::Int)::Array{Int,2}

Returns all the possible `q^len` strings of length `len` with values in 𝔾𝔽(q) as
columns of a matrix
"""
function allgfqstrings(q::Int, len::Int)::Array{Int,2}
    if len > 10
        @warn "Exponential alert! This operation requires $q^$len operations"
    end
    return hcat([digits(j, base=q, pad=len) for j = 0:q^len-1]...)
end

"""
Exhaust_enum <: LossyAlgo

Algorithm to solve lossy compression problem exactly by exhaustive enumeration.
"""
struct ExhaustEnum <: LossyAlgo; end

function solve!(lm::LossyModel, algo::ExhaustEnum)
    solutions = enum_solutions(lm)
    distortions = [distortion(lm,x) for x in solutions]
    (minval, minidx) = findmin(distortions)
    lm.x = solutions[minidx]
    return (:foundExactSol, distortion(lm))
end


"""
solutions_distances(V::Vector{Vector{Int}}; cutoff::Real=Inf)

Computes the matrix D of pairwise distances between 𝔾𝔽(2ᵏ) vectors
contained in `V`.
Optionally set to zero all distances greater or equal than `cutoff`
"""
function solutions_distances(V::Vector{Vector{Int}}; cutoff::Real=Inf)
    D = [hd(V[i],V[j])*(hd(V[i],V[j])<cutoff) for i=eachindex(V), j=eachindex(V)]
end

function solutions_distances(lm::LossyModel; kwargs...)
    return solutions_distances(enum_solutions(lm); kwargs...)
end


import LightGraphs.connected_components
"""
    connected_components(lm::LossyModel; kwargs...)

Returns the connected components of the graph where nodes are solutions and
there are edges between nodes weigthed by their (Hamming) distance if is less
than `cutoff`.
"""
function connected_components(lm::LossyModel; kwargs...)
    distances_graph = solutions_graph(lm; kwargs...)
    return LightGraphs.connected_components(distances_graph)
end

"""
    n_cc(lm::LossyModel; kwargs...)

Returns the number of connected components
"""
function n_cc(lm::LossyModel; kwargs...)
    length(connected_components(lm; kwargs...))
end

import SimpleWeightedGraphs.SimpleWeightedGraph
"""
    solutions_graph(lm::LossyModel; kwargs...)

Returns a `SimpleWeightedGraph` object containing the graph where nodes are
solutions and there are edges between nodes weigthed by their (Hamming) distance
if is less than `cutoff`.
"""
function solutions_graph(lm::LossyModel; kwargs...)
    d = solutions_distances(lm; kwargs...)
    return SimpleWeightedGraphs.SimpleWeightedGraph(d)
end


"""
    plot_solutions_graph(lm::LossyModel; kwargs...)

Plots the graph built by solutions and their distances
"""
function plot_solutions_graph(lm::LossyModel; layout::Symbol=:stress,
    kwargs...)
    g = solutions_graph(lm; kwargs...)
    nvertices = LightGraphs.nv(g)
    # nodelabel_vec = int2gfq(collect(0:nvertices-1), Int(log2(lm.fg.q)))
    # nodelabel = [join(string.(v)) for v in nodelabel_vec]
    nodelabel = 1:nvertices
    D = solutions_distances(lm; kwargs...)
    edgelabel = Dict((i,j)=>D[i,j] for j=2:nvertices for i=1:j-1 if D[i,j]!=0)
    p = graphplot(g, curves=false,
        names=nodelabel, nodeshape=:circle, nodesize=0.2, nodecolor=:lightblue,
        method=layout, fontsize=7, 
        edgelabel=edgelabel,
        edgelabelbox=true)
    return p
end


"""
    wef(lm::LossyModel)

Computes the weight enumeration function: number of solutions at any distance
from the zero vector
"""
function wef(lm::LossyModel)
    dist = solutions_distances(lm)
    dist_from_zero = dist[:,1]
    max_dist = lm.fg.n
    return StatsBase.counts(dist_from_zero,max_dist)
end

function plot_wef(lm::LossyModel)
    w = wef(lm)
    p = Plots.bar(w, label="", title="WEF")
    # println("Total number of solutions: ", sum(w)+1)
    return p
end



