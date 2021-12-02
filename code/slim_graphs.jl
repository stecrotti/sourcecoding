using SparseArrays, LinearAlgebra, ProgressMeter

function readgraph(graph)
    I=Int[]
    J=Int[]
    open(graph) do f
        linecounter = 0
        for l in eachline(f)
            v = split(l)
            if v[1] == "D"
                i,j = extrema((parse(Int, v[2])+1, parse(Int, v[3])+1))
                push!(I, i)
                push!(J, j)
            end 
        end
    end

    N = maximum(I)
    H = sparse(J.-N,I,fill(1,length(I))) .÷ 2
end

function readseeds(seeds, H)
    M, N = size(H)
    @show size(H)
    v = Int[]
    open(seeds) do f
        for l in eachline(f)
            if l[1] != '#'
                v = parse.(Int, split(l)) .- N .+ 1
            end
        end
        println("$(length(v)) removed factors $v")
        return H[setdiff(1:size(H,1), v), :]
    end
end

function leaf_removal(H::SparseMatrixCSC, Ht = permutedims(H);
        degs = vec(sum(H .!= 0, dims=1)))
    degs .= vec(sum(H .!= 0, dims=1))
    M, N = size(H)
    facts = trues(M)
    rowperm = Int32[]
    Q = Int32.(findall(degs .== 1))
    indep = Int32.(findall(degs .== 0))
    dep = Int32[]
    sizehint!(indep, N-M)
    sizehint!(dep, M)
    sizehint!(rowperm, M)
    while !isempty(Q)
        i = popfirst!(Q)
        degs[i] == 0 && continue
        push!(dep, i)
        ∂i = @view rowvals(H)[nzrange(H,i)]
        a = ∂i[findfirst(b->facts[b], ∂i)]
        facts[a] = false
        push!(rowperm, a) 
        for j in @view rowvals(Ht)[nzrange(Ht,a)]
            degs[j] -= 1
            if j != i
                if degs[j] == 0
                    push!(indep, j)
                elseif degs[j] == 1
                    push!(Q, j)
                end
            end
        end
    end
    all(degs .==  0) || @warn "non-empty core"
    rowperm, dep, indep
end

isuppertriang(H::SparseMatrixCSC) = all(rowvals(H)[last(nzrange(H,i))] == i for i = 1:size(H,1))

function ut2diagGF2!(T::SparseMatrixCSC)
    (m,n) = size(T)
    # Check that the left part of T is unit upper triangular
    @assert isuppertriang(T)
    # Loop over diagonal elements
    for c in m:-1:2
        # Find non-zero elements above T[c,c] and perform row operations to 
        #  cancel them out
        for (j,v) in @views zip(rowvals(T[:,c]),nonzeros(T[:,c]))
            if v != 0 && j < c
                for k in rowvals(T[c,c:end]).+c.-1
                    T[j,k] ⊻= T[c,k]
                end
            end
        end
    end
    dropzeros!(T)
end

function ut2diagGFQ!(T::SparseMatrixCSC, q::Int; gftab = gftables(Val(q)))
    mult, gfdiv = gftab
    m, n = size(T)
    # Check that the left part of T is upper triangular
    @assert isuppertriang(T)
    # Loop over diagonal elements
    for c in m:-1:1
        # Normalize the c-th row so that T[c,c]=1
        Tcc = T[c,c] + 1
        # normalize row `p` so that pivot equal 1
        for k in rowvals(T[c,c:end]).+c.-1
            T[c,k] = gfdiv[T[c,k]+1, Tcc] - 1
        end
        # Find non-zero elements above T[c,c] and perform row operations to 
        #  cancel them out
        for (j,v) in @views zip(rowvals(T[:,c]),nonzeros(T[:,c]))
            if v != 0 && j < c
                for k in rowvals(T[c,c:end]).+c.-1
                    T[j,k] = xor( T[j,k], mult[v+1,T[c,k]+1]-1 ) 
                end
            end
        end
    end
    dropzeros!(T)
end

function ut2diagGF2(Tdep, Tindep)
    m = size(Tdep,1)
    @assert isuppertriang(Tdep)
    # Store the rows of the right part in a vector of sparse vectors
    Tt = permutedims(Tindep)
    R = [Tt[:,r] for r in 1:m]
    # Loop over diagonal elements
    @inbounds for c in m:-1:2
        # Loop over the elements above T[c,c]
        for j in @view rowvals(Tdep)[nzrange(Tdep,c)[1:end-1]]
            R[j] .⊻= R[c]
        end
    end
    #U = reduce(hcat,R)
    R
end

function ut2diagGF2(Hdep, Hindep, rowperm)
    m = length(rowperm)
    invrowperm = invperm(rowperm)
    # Store the rows of the right part in a vector of sparse vectors
    Hindep_t = permutedims(Hindep)
    R = [Hindep_t[:,r] for r in rowperm]
    # Loop over diagonal elements
    @inbounds for c in m:-1:2
        # Loop over the elements above T[c,c]
        for j in @view invrowperm[rowvals(Hdep)[nzrange(Hdep,c)]]
            j!=c && (R[j] .⊻= @view R[c])
        end
    end
    #U = reduce(hcat,R)
    R
end

function echelonize(H, Ht, rowperm, colperm)
    m = size(H,1)
    # @assert isuppertriang(H[rowperm, colperm])
    # Store the rows of the right part in a vector of sparse vectors
    R = [Ht[:,r] for r in 1:m]
    # Loop over diagonal elements
    for c in m:-1:2
        # Loop over the elements above T[c,c]
        for j in @view rowvals(H)[nzrange(H,colperm[c])]
            if j ≠ rowperm[c]
                R[j] .⊻= R[rowperm[c]]
            end
        end
    end
    R
end

function findbasis(H, Ht = permutedims(H))
    rowperm, dep, indep = leaf_removal(H, Ht)
    colperm = [dep; indep]
    Hnew = H[rowperm, colperm]
    ut2diagGF2!(Hnew)
    B = [Hnew[:, size(Hnew, 1)+1:end]; I]
    indep = colperm[size(H,1)+1:end]
    B[invperm(colperm),:], indep
end 

function findbasis(H, q::Int, Ht = permutedims(H); gftab = gftables(Val(q)))
    rowperm, dep, indep = leaf_removal(H, Ht)
    colperm = [dep; indep]
    Hnew = H[rowperm, colperm]
    ut2diagGFQ!(Hnew, q; gftab = gftab)
    B = [Hnew[:, size(Hnew, 1)+1:end]; I]
    indep = colperm[size(H,1)+1:end]
    B[invperm(colperm),:], indep
end

function gfrrefGF2!(H::AbstractArray{<:Number,2})
    (m,n) = size(H)
    # Initialize pivot to zero
    dep = Int[]
    p = 0
    for c = 1:n
        nz = findfirst(!iszero, H[p+1:end,c])
        if nz === nothing
            continue
        else
            p += 1
            push!(dep, c)
            if nz != 1
                H[p,:], H[p+nz-1,:] = H[nz+p-1,:], H[p,:]
            end
            # Apply row-wise xor to rows above and below the pivot
            for r = [1:p-1; p+1:m]
                if H[r,c] != 0
                    for cc in c:n
                        H[r,cc] = xor(H[r,cc], H[p,cc])
                    end
                end
            end
            if p == m 
                break
            end
        end
    end
    return H, dep
end
gfrrefGF2(H::AbstractArray{<:Number,2}) = gfrrefGF2!(copy(H))

function findbasis_slow(H)
    A,dep = gfrrefGF2(H)
    indep = setdiff(1:size(H,2), dep)
    colperm = [dep; indep]
    B = [A[1:length(dep),indep];I]
    B .= B[invperm(colperm),:]
    B, indep
end

function gfrcefGF2!(H::AbstractArray{<:Number,2})
    (n,m) = size(H)
    # Initialize pivot to zero
    dep = Int[]
    p = 0
    for r = 1:n
        nz = findfirst(!iszero, H[r,p+1:end])
        if nz === nothing
            continue
        else
            p += 1
            push!(dep, r)
            if nz != 1
                H[:,p], H[:,p+nz-1] = H[:,p+nz-1], H[:,p]
            end
            # Apply colum-wise xor to rows left and right of the pivot
            for c = [1:p-1; p+1:m]
                if H[r,c] != 0
                    for rr in r:n
                        H[rr,c] = xor(H[rr,c], H[rr,p])
                    end
                end
            end
            if p == m 
                break
            end
        end
    end
    return H, dep
end
gfrcefGF2(H::AbstractArray{<:Number,2}) = gfrcefGF2!(copy(H))
