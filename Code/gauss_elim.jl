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


function gfref!(H::Array{Int,2},
                q::Int=2,
                mult::OffsetArray{Int,2}=OffsetArray(zeros(Int,q,q), 0:q-1, 0:q-1))

    !ispow(H, 2) && error("q must be a power of 2")
    !isgfq(H, q) && error("Matrix H has values outside GF(q)")

end

function ispow(x::Int, b::Int)
    if x > 0
        return isinteger(log(x,b))
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
