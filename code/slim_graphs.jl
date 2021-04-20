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

function leaf_removal(H::SparseMatrixCSC)
    M, N = size(H)
    Ht = sparse(transpose(H)) # for faster access by row
    degs = vec(sum(H .!= 0, dims=1))
    facts = fill(true, M)
    rowperm = Int[]
    Q = findall(degs .== 1)
    indep = findall(degs .== 0)
    dep = Int[]
    while !isempty(Q)
        i = popfirst!(Q)
        degs[i] == 0 && continue
        push!(dep, i)
        ∂i = rowvals(H)[nzrange(H,i)]
        ∂i = ∂i[facts[∂i]]
        @assert length(∂i) == 1 # should be a residual leaf
        a = ∂i[1]
        facts[a] = false
        push!(rowperm, a) 
        for j in rowvals(Ht)[nzrange(Ht,a)]
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
    rowperm, [dep; indep]
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

function findbasis(H)
    rowperm, colperm = leaf_removal(H)
    Hnew = H[rowperm, colperm]
    ut2diagGF2!(Hnew)
    B = [Hnew[:, size(Hnew, 1)+1:end]; I]
    indep = colperm[size(H,1)+1:end]
    B[invperm(colperm),:], indep
end 

function gfrrefGF2!(H::AbstractArray{<:Number,2})
    (m,n) = size(H)
    # Initialize pivot to zero
    dep = Int[]
    p = 0
    @showprogress for c = 1:n
        nz = findfirst(!iszero, H[p+1:end,c])
        if nz === nothing
            continue
        else
            p += 1
            push!(dep, c)
            if nz != 1
                H[p,:], H[nz+p-1,:] = H[nz+p-1,:], H[p,:]
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
    A,dep = gfrrefGF2!(H)
    indep = setdiff(1:size(H,2), dep)
    colperm = [dep; indep]
    B = [A[1:length(dep),indep];I]
    B .= B[invperm(colperm),:]
    B, indep
end