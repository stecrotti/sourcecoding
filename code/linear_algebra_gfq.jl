#### Linear algebra on 𝔾𝔽(2ᵏ) ####
using LinearAlgebra, LightGraphs, SimpleWeightedGraphs, SparseArrays


"""Reduce matrix over GF(q) to reduced row echelon form"""
function gfrref!(H::Array{Int,2},
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
            H[p:end,:] .= sortslices(H[p:end,:], dims=1, rev=true)
            # Normalize row of the pivot to make it 1
            H[p,:] .= gfdiv[H[p,:], H[p,c]]
            # Apply row-wise xor to rows below the pivot
            for r = vcat([1:p-1, p+1:m]...)
                if H[r,c] != 0
                    # Adjust to make pivot 1
                    f = gfdiv[H[p,c], H[r,c]]
                    H[r,:] .= xor.(gfmult[f, H[r,:]], H[p,:])
                # else
                #     # From now on only zeros in column c
                #     break
                end
            end
            p == m && break
        end
    end
    return H
end

function gfrcef!(H::Array{Int,2},
                q::Int=2,
                gfmult::OffsetArray{Int,2}=gftables(q)[1],
                gfdiv::OffsetArray{Int,2}=gftables(q)[3])
    H .= permutedims(gfrref(permutedims(H), q, gfmult, gfdiv))
end

function gfrref(H::Array{Int,2},
                q::Int=2,
                gfmult::OffsetArray{Int,2}=gftables(q)[1],
                gfdiv::OffsetArray{Int,2}=gftables(q)[3])
    tmp = copy(H)
    gfrref!(tmp, q, gfmult, gfdiv)
    return tmp
end

function gfrcef(H::Array{Int,2},
                q::Int=2,
                gfmult::OffsetArray{Int,2}=gftables(q)[1],
                gfdiv::OffsetArray{Int,2}=gftables(q)[3])
    tmp = copy(H)
    gfrcef!(tmp, q, gfmult, gfdiv)
    return tmp
end

# Convert upper triangular matrix into diagonal
#  [T|X] -> [D|Y] 
function ut2diag!(T::Array{Int,2}, q::Int=2,
        gfmult::OffsetArray{Int,2}=gftables(q)[1],
        gfdiv::OffsetArray{Int,2}=gftables(q)[3])
        
    (m,n) = size(T)
    # Check that the left part of T is unit upper triangular
    @assert isunituppertriangular(T[:,1:m])
    # Loop over diagonal elements
    for c in m:-1:1
        # Find non-zero elements above T[c,c] and perform row operations to 
        #  cancel them out
        nz = (T[1:c-1,c] .!= 0)
        for j in findall(nz)
            T[j,c:end] .= 
                xor.(T[j,c:end], gfmult[gfdiv[T[j,c],T[c,c]], T[c,c:end]])
        end
    end
    return T
end
ut2diag(H::Array{Int,2}, args...) = ut2diag!(copy(H), args...)

n_nonzerorows(H::Array{Int,2}) = sum([!all(H[r,:] .== 0) for r in 1:size(H,1)])

function gfrank(H::Array{Int,2}, q::Int=2,
                gfmult::OffsetArray{Int,2}=gftables(q)[1],
                gfdiv::OffsetArray{Int,2}=gftables(q)[3])
    # Reduce to row echelon form
    Href = gfrref(H, q, gfmult, gfdiv)
    return n_nonzerorows(Href)
end

function gfnullspace(H::Array{Int,2}, q::Int=2,
                gfmult::OffsetArray{Int,2}=gftables(q)[1],
                gfdiv::OffsetArray{Int,2}=gftables(q)[3])
    nrows,ncols = size(H)
    dimker = ncols - gfrank(H, q, gfmult, gfdiv)
    # As in https://en.wikipedia.org/wiki/Kernel_(linear_algebra)#Computation_by_Gaussian_elimination
    HI = [H; I]
    gfrcef!(HI, q)
    ns = HI[nrows+1:end, end-dimker+1:end]
    return ns
end

# Works only for GF(2^k)
function gfdot(x::Vector{Int}, y::Vector{Int}, q::Int,
    mult::OffsetArray{Int,2,Array{Int,2}}=gftables(q)[1])
    L = length(x)
    @assert length(y) == L
    return reduce(xor, [mult[x[k],y[k]] for k=1:L], init=0)
end

# Works only for GF(2^k)
function gfmatrixmult(A::Array{Int,2}, B::Array{Int,2}, q::Int=2,
    mult::OffsetArray{Int,2,Array{Int,2}}=gftables(q)[1])
    m,n = size(A)
    r,p = size(B)
    q,s = size(mult)
    @assert r == n  # check compatibility of dimensions for matrix product
    @assert q == s  # check valid multiplication matrix
    C = zeros(Int, m, p)
    for j = 1:p
        for i = 1:m
            C[i,j] = gfdot(A[i,:], B[:,j], q, mult)
        end
    end
    # If C is a column vector, just return it as a 1D Array
    p==1 && (C = vec(C))
    return C
end
function gfmatrixmult(A::Array{Int,2}, V::Vector{Int}, args...)
    return gfmatrixmult(A, hcat(V))
end

function gfmatrixinv(H::Array{Int,2},
        q::Int=2,
        gfmult::OffsetArray{Int,2}=gftables(q)[1],
        gfdiv::OffsetArray{Int,2}=gftables(q)[3])
    @assert issquare(H)
    n = size(H,1)
    augmented = gfrref(hcat(A,I), q, gfmult, gfdiv)
    if n_nonzerorows(augmented[:,1:n]) < n
        error("Matrix to be inverted is not full rank")
    else
        return augmented[:,n+1:end]
    end
end

function gf_invert_ut(T::Array{Int,2}, y::Vector{Int},
    q::Int=2,
    gfmult::OffsetArray{Int,2}=gftables(q)[1],
    gfdiv::OffsetArray{Int,2}=gftables(q)[3])

    @assert issquare(T)
    @assert isunituppertriangular(T)
    n = size(T,1)
    @assert n==length(y)
    @assert isgfq(T,q) && isgfq(y,q)

    x = zeros(Int, n)
    for k in n:-1:1
        x[k] = gfdiv[xor(y[k], reduce(xor, gfmult[T[k,i],x[i]] for i=k:n)), T[k,k]]
    end
    return x
end
function gf_invert_ut(T::Array{Int,2}, y::Array{Int,2}, args...)
    @assert size(T,1)==size(y,1)
    x = hcat([gf_invert_ut(T,y[:,c],args...) for c in 1:size(y,2)]...)
    return x
end

###### UTILS ######

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
function isuppertriangular(A::Array{Int,2})
    return A == UpperTriangular(A)
end
function isunituppertriangular(A::Array{Int,2})
    return isuppertriangular(A) && all(diag(A).==1)
end
function issquare(A::Array{Int,2})
    (m,n) = size(A)
    return m==n
end
