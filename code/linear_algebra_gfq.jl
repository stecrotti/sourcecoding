#### Linear algebra on 𝔾𝔽(2ᵏ) ####
using LinearAlgebra, LightGraphs, SimpleWeightedGraphs, SparseArrays


"""Reduce matrix over GF(q) to reduced row echelon form"""
function gfrref!(H::AbstractArray{Int,2},
                q::Int=2,
                gfmult::OffsetArray{Int,2}=gftables(q)[1],
                gfdiv::OffsetArray{Int,2}=gftables(q)[3];
                column_perm::AbstractVector=zeros(Int, size(H,2)))

    !ispow(q, 2) && error("q must be a power of 2")
    !isgfq(H, q) && error("Matrix H has values outside GF(q) with q=$q")
    (m,n) = size(H)
    # Initialize pivot to zero
    pivot_column_indices = zeros(Int, m)
    p = 0
    @showprogress for c = 1:n
        if iszero(H[p+1:end,c])
            continue
        else
            p += 1
            pivot_column_indices[p] = c
            # sort rows of H so that all zeros in the c-th column are at the bottom
            H[p:end,:] .= sortslices(H[p:end,:], dims=1, rev=true)
            # Normalize row of the pivot to make it 1
            H[p,:] .= gfdiv[H[p,:], H[p,c]]
            # Apply row-wise xor to rows below the pivot
            for r = [1:p-1; p+1:m]
                if H[r,c] != 0
                    f = gfdiv[H[r,c], H[p,c]]
                    H[r,:] .= xor.(gfmult[f, H[p,:]], H[r,:])
                # else
                #     # From now on only zeros in column c
                #     break
                end
            end
            if p == m 
                break
            end
        end
    end
    # Permute columns to get an identity matrix on the left
    column_perm .= [pivot_column_indices[1:p]; setdiff(1:n, pivot_column_indices[1:p])]
    return H
end

function gfrcef!(H::AbstractArray{Int,2},
                q::Int=2,
                gfmult::OffsetArray{Int,2}=gftables(q)[1],
                gfdiv::OffsetArray{Int,2}=gftables(q)[3];
                row_perm::AbstractVector=zeros(Int,size(H,1)))
    H .= permutedims(gfrref(permutedims(H), q, gfmult, gfdiv, column_perm=row_perm))
end

function gfrref(H::AbstractArray{Int,2},
                q::Int=2,
                gfmult::OffsetArray{Int,2}=gftables(q)[1],
                gfdiv::OffsetArray{Int,2}=gftables(q)[3]; kw...)
    tmp = copy(H)
    gfrref!(tmp, q, gfmult, gfdiv; kw...)
    return tmp
end

function gfrcef(H::AbstractArray{Int,2},
                q::Int=2,
                gfmult::OffsetArray{Int,2}=gftables(q)[1],
                gfdiv::OffsetArray{Int,2}=gftables(q)[3]; kw...)
    tmp = copy(H)
    gfrcef!(tmp, q, gfmult, gfdiv; kw...)
    return tmp
end

# Convert upper triangular matrix into diagonal
#  [T|X] -> [D|Y] 
function ut2diag!(T::AbstractArray{Int,2}, q::Int=2,
        gfmult::OffsetArray{Int,2}=gftables(q)[1],
        gfdiv::OffsetArray{Int,2}=gftables(q)[3])
        
    (m,n) = size(T)
    # Check that the left part of T is unit upper triangular
    @assert isuppertriangular(T[:,1:m])
    # If not only 1's on the diagonal -> do row operations
    if !isunituppertriangular(T[:,1:m])
        for k in 1:m
            T[k,:] .= gfdiv[T[k,:], T[k,k]]
        end
    end
    # Loop over diagonal elements
    for c in m:-1:2
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
ut2diag(H::AbstractArray{Int,2}, args...) = ut2diag!(copy(H), args...)

n_nonzerorows(H::AbstractArray{Int,2}) = sum(!all(H[r,:] .== 0) for r in 1:size(H,1))

function gfrank(H::AbstractArray{Int,2}, q::Int=2,
                gfmult::OffsetArray{Int,2}=gftables(q)[1],
                gfdiv::OffsetArray{Int,2}=gftables(q)[3])
    # Reduce to row echelon form
    Href = gfrref(H, q, gfmult, gfdiv)
    return n_nonzerorows(Href)
end

function gfnullspace(H::AbstractArray{Int,2}, q::Int=2,
                gfmult::OffsetArray{Int,2}=gftables(q)[1],
                gfdiv::OffsetArray{Int,2}=gftables(q)[3];
                column_perm::AbstractVector=zeros(Int,size(H,2)))
    nrows,ncols = size(H)
    # Reduce to rref
    Hrref = gfrref(H, q, gfmult, gfdiv; column_perm=column_perm)
    # Re-arrange columns to get a form [I | U]
    Hrref .= Hrref[:,column_perm]
    # Build basis
    rk = n_nonzerorows(Hrref)
    B = [Hrref[1:rk,rk+1:end]; I]
    # Restore original column order
    B .= B[invperm(column_perm),:]
    return B
end

# Works only for GF(2^k)
function gfdot(x::AbstractVector{Int}, y::AbstractVector{Int}, q::Int,
    mult::OffsetArray{Int,2,Array{Int,2}}=gftables(q)[1])
    L = length(x)
    @assert length(y) == L
    return reduce(xor, [mult[x[k],y[k]] for k=1:L], init=0)
end

# Works only for GF(2^k)
function gfmatrixmult(A::AbstractArray{Int,2}, B::AbstractArray{Int,2}, q::Int=2,
        mult::OffsetArray{Int,2,Array{Int,2}}=gftables(q)[1])
    m,n = size(A)
    r,p = size(B)
    q,s = size(mult)
    @assert r == n  # check compatibility of dimensions for matrix product
    @assert q == s  # check valid multiplication matrix
    C = zeros(Int, m, p)
    @inbounds for j = 1:p
        for i = 1:m
            C[i,j] = gfdot(A[i,:], B[:,j], q, mult)
        end
    end
    return C
end

function gfmatrixmult(A::AbstractArray{Int,2}, b::AbstractVector{Int}, q::Int=2,
        gfmult::OffsetArray{Int,2,Array{Int,2}}=gftables(q)[1])
    m,n = size(A)
    r = length(b)
    @assert r == n  # check compatibility of dimensions for matrix product
    if q > 2
        C = [gfdot(A[i,:], b, q, gfmult) for i in 1:m]
    else
        C = (A * b) .% 2
    end
    return C
end


function gfmatrixinv(H::AbstractArray{Int,2},
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

function gf_invert_ut(T::AbstractArray{Int,2}, y::AbstractVector{Int},
    q::Int=2,
    gfmult::OffsetArray{Int,2}=gftables(q)[1],
    gfdiv::OffsetArray{Int,2}=gftables(q)[3], x = zeros(Int, n))

    @assert issquare(T)
    @assert isuppertriangular(T)
    n = size(T,1)
    @assert n==length(y)
    @assert isgfq(T,q) && isgfq(y,q)

    for k in n:-1:1
        x[k] = gfdiv[xor(y[k], reduce(xor, gfmult[T[k,i],x[i]] for i=k:n)), T[k,k]]
    end
    return x
end
function gf_invert_ut(T::AbstractArray{Int,2}, y::AbstractArray{Int,2}, args...)
    @assert size(T,1)==size(y,1)
    x = reduce(hcat, [gf_invert_ut(T,y[:,c],args...) for c in 1:size(y,2)])
    return x
end

# Move between GF(2) and GF(2^k)
function system_gfqto2(H::AbstractArray{Int,2},
    q_and_tables...; getbasis::Function=gfnullspace,
    H2::SparseMatrixCSC{Int,Int}=spzeros(Int, ))
    # Find a basis for the solutions of H
    q = q_and_tables[1]
    B = getbasis(H, q_and_tables...)
    # Check that #rows of H is an integer multiple of k
    @assert mod(size(H,1), Int(log2(q)))==0
    # Convert each column of B (each basis vector) to GF(2)
    B2 = hcat([gfqto2(B[:,j],Int(log2(q))) for j in 1:size(B,2)]...)
    #  B2 is now a GF(2) basis
    # Build a corresponding parity-check matrix
    Hgf2 = basis2matrix(B2, q_and_tables..., H=H2)
    return Hgf2
end

# Given a basis for the space of solutions stored as columns of `B`,
#  build a parity-check matrix whose kernel is the space of solutions
# H is passed as argument to avoid allocating
function basis2matrix(B::AbstractArray{Int,2},
    q_and_tables...;
    H::SparseMatrixCSC{Int,Int}=spzeros(Int, size(B,1)-size(B,2), size(B,1)))
    # Reduce B to reduced column echelon form [I;A]
    Brcef = gfrcef(B, q_and_tables...)
    # Build H as [A I]
    m,n = size(H)
    H[diagind(H,n-m)] .= 1
    H[:,1:n-m] = Brcef[n-m+1:end,:]
    return H
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
function isinspan(A::AbstractArray{Int,2}, v::AbstractVector{Int}; q::Int=2)
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
function isuppertriangular(A::AbstractArray{Int,2})
    return A == UpperTriangular(A)
end
function isunituppertriangular(A::AbstractArray{Int,2})
    return isuppertriangular(A) && all(diag(A).==1)
end
function issquare(A::AbstractArray{Int,2})
    (m,n) = size(A)
    return m==n
end
