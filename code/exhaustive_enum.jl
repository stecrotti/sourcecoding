function enum_solutions(H::Array{Int,2}, q::Int=2)
    !ispow(q, 2) && error("q must be a power of 2")
    !isgfq(H, q) && error("Matrix H has values outside GF(q) with q=$q")

    (m,n) = size(H)
    # Basis for the space of solutions
    ns = gfnullspace(H, q)
    k = size(ns,2)  # Nullspace dimension
    # All possible 2ᵐ coefficient combinations
    coeffs = hcat([digits(j, base=q, pad=k) for j = 0:2^k-1]...)
    # Multiply to get all possible linear combinations of basis vectors
    return gfmatrixmult(ns, coeffs, q)
end

enum_solutions(fg::FactorGraph) = enum_solutions(adjmat(fg), fg.q)
enum_solutions(lm::LossyModel) = enum_solutions(lm.fg)
