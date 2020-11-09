#### Ranks, kernels and gaussian elimination on 𝔾𝔽(2ᵏ) ####

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
            @show c,p
            H[p:end,:] .= sortslices(H[p:end,:], dims=1, rev=true,
                lt=(row1,row2)->row1[c]==0)
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
