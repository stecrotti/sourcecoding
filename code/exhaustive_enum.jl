using LightGraphs, SimpleWeightedGraphs

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
    allgfqstrings(q::Int, k::Int)::Array{Int,2}

Returns all the possible qᵏ strings of length k with values in 𝔾𝔽(q) as
columns of a matrix
"""
function allgfqstrings(q::Int, k::Int)::Array{Int,2}
    return hcat([digits(j, base=q, pad=k) for j = 0:q^k-1]...)
end


"""
    pairwise_distances(V::Vector{Vector{Int}}; cutoff::Int=0)

Computes the matrix D of pairwise distances between 𝔾𝔽(2ᵏ) vectors
contained in `V`. 
Optionally set to zero all distances greater than `cutoff`
"""
function pairwise_distances(V::Vector{Vector{Int}}; cutoff::Int=0)
    D = [hd(V[i],V[j])*(hd(V[i],V[j])>cutoff) for i=eachindex(V), j=eachindex(V)]
end

function pairwise_distances(lm::LossyModel; kwargs...)
    pairwise_distances(enum_solutions(lm); kwargs...)
end

"""
    Enum <: LossyAlgo

Algorithm to solve lossy compression problem exactly by exhaustive enumeration.
"""
struct Enum <: LossyAlgo; end

function solve!(lm::LossyModel, algo::Enum)
    solutions = enum_solutions(lm)
    distortions = [distortion(lm,x) for x in solutions]
    (minval, minidx) = findmin(distortions)
    lm.x = solutions[minidx]
    return (:foundExactSol, distortion(lm))
end