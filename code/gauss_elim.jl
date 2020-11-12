#### Ranks, kernels and gaussian elimination on 𝔾𝔽(2ᵏ) ####
using LinearAlgebra, LightGraphs, SimpleWeightedGraphs

"""Reduce matrix over GF(q) to row echelon form"""
function gfref!(H::Array{Int,2},
                q::Int=2,
                gfmult::OffsetArray{Int,2}=gftables(q)[1],
                gfdiv::OffsetArray{Int,2}=gftables(q)[3])

    !ispow(q, 2) && error("q must be a power of 2")
    !isgfq(H, q) && error("Matrix H has values outside GF(q) with q=$q")
    (m,n) = size(H)
    # Initialize pivot to zero
    p = 0
    for c = 1:n
        if iszero(H[p+1:end,c])
            continue
        else
            p += 1
            # sort rows of H so that all zeros in the c-th column are at the bottom
            # H[p:end,:] .= sortslices(H[p:end,:], dims=1, rev=true,
            #     lt=(row1,row2)->row1[c]==0)
            H[p:end,:] .= sortslices(H[p:end,:], dims=1, rev=true)
            # Normalize row of the pivot to make it 1
            H[p,:] .= gfdiv[H[p,:], H[p,c]]
            # Apply row-wise xor to rows below the pivot
            for r = p+1:m
                if H[r,c] != 0
                    # Adjust to make pivot 1
                    f = gfdiv[H[p,c], H[r,c]]
                    H[r,:] .= xor.(gfmult[f, H[r,:]], H[p,:])
                end
            end
            p == m && break
        end
    end
end

function gfcef!(H::Array{Int,2},
                q::Int=2,
                gfmult::OffsetArray{Int,2}=gftables(q)[1],
                gfdiv::OffsetArray{Int,2}=gftables(q)[3])
    H .= permutedims(gfref(permutedims(H), q, gfmult, gfdiv))
end

function gfref(H::Array{Int,2},
                q::Int=2,
                gfmult::OffsetArray{Int,2}=gftables(q)[1],
                gfdiv::OffsetArray{Int,2}=gftables(q)[3])
    tmp = copy(H)
    gfref!(tmp, q, gfmult, gfdiv)
    return tmp
end

function gfcef(H::Array{Int,2},
                q::Int=2,
                gfmult::OffsetArray{Int,2}=gftables(q)[1],
                gfdiv::OffsetArray{Int,2}=gftables(q)[3])
    tmp = copy(H)
    gfcef!(tmp, q, gfmult, gfdiv)
    return tmp
end

function gfrank(H::Array{Int,2}, q::Int=2,
                gfmult::OffsetArray{Int,2}=gftables(q)[1],
                gfdiv::OffsetArray{Int,2}=gftables(q)[3])
    # Reduce to row echelon form
    Href = gfref(H, q, gfmult, gfdiv)
    # Count number of all-zero rows
    nonzero = [!all(Href[r,:] .== 0) for r in 1:size(H,1)]
    # nonzero = reduce(, Href, dims=2)
    # Sum
    return sum(nonzero)
end

function gfnullspace(H::Array{Int,2}, q::Int=2,
                gfmult::OffsetArray{Int,2}=gftables(q)[1],
                gfdiv::OffsetArray{Int,2}=gftables(q)[3])
    nrows,ncols = size(H)
    dimker = ncols - gfrank(H, q, gfmult, gfdiv)
    # As in https://en.wikipedia.org/wiki/Kernel_(linear_algebra)#Computation_by_Gaussian_elimination
    HI = [H; I]
    HIcef = gfcef(HI, q)
    ns = HIcef[nrows+1:end, end-dimker+1:end]
    return ns
end

function ispow(x::Int, b::Int)
    if x > 0
        return isinteger(log(b,x))
    else
        return false
    end
end

function isgfq(X, q::Int)
    for x in X
        if (x<0 || x>q-1 || !isinteger(x))
            return false
        end
    end
    return true
end

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

function isinspan(A::Array{Int,2}, v::Vector{Int}; q::Int=2)
    if isempty(A)
        return true
    else
        @assert length(v) == size(A,1)
        rkA = gfrank(A, q)
        Av = [A v]
        rkAv = gfrank(Av)
        return rkA == rkAv
    end
end

###### OLD STUFF #######

function gf2ref!(H::BitArray{2})
    (m,n) = size(H)

    # Initialize pivot to zero
    p = 0

    for c = 1:n
        if iszero(H[p+1:end,c])
            continue
        else
            p += 1
            # sort rows of H so that all zeros in the c-th column are at the bottom
            H[p:end,:] .= sortslices(H[p:end,:], dims=1, rev=true,
                lt=(row1,row2)->row1[c]==0)
            # Apply row-wise xor to rows below the pivot
            for r = p+1:m
                if H[r,c] == true
                    H[r,:] .= xor.(H[r,:], H[p,:])
                end
            end
            if p == m-1
                break
            end
        end
        println("c = $c")
        display(H)
    end
end

function gf2ref(H::BitArray{2})
    tmp = copy(H)
    gf2ref!(tmp)
    return tmp
end
