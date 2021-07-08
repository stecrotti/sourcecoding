include("bp_full.jl")
using OffsetArrays

msg2var(msg_idx) = div.(msg_idx.+1, 2) 
msg2fact(msg_idx, H) = rowvals(H)[msg_idx]
msg2field(m::Tuple) = (m[1]-m[2])/2
spin2bool(x::Int) = x.== -1
bool2spin(x::Bool) = (-1) .^ x


function comp_tree(H::SparseMatrixCSC, X::SparseMatrixCSC, 
        u, h, efield::Vector{Tuple{T,T}}, 
        root::Int, root_belief::Tuple{T,T}, σ_gs::Vector{Int}) where {T<:Real}
    
    k = length(u)-1
    P = fill(0, -k:k)
    P[0] = root
    b = fill((zero(T),zero(T)), -k:k)
    b[0] = root_belief

    factor_neigs = [nonzeros(X)[nzrange(X, a)] for a = 1:size(H,1)]
    variable_neigs = [∂i = nzrange(H, i) for i = 1:size(H,2)]

    a_left, a_right = msg2fact(variable_neigs[root], H)

    for t in 1:k-1
        # go right
        i = P[t-1]
        ∂a_right = factor_neigs[a_right]
        j, bnext, a_right = update_layer(H, X, variable_neigs, σ_gs, efield, b[t-1], 
            ∂a_right, i, a_right, u[k-t], u[k-t-1], h[k-t])
        P[t] = j
        b[t] = bnext
        # go left
        i = P[-(t-1)]
        ∂a_left = factor_neigs[a_left]
        j, bnext, a_left = update_layer(H, X, variable_neigs,σ_gs, efield, 
            b[-(t-1)], ∂a_left, i, a_left, u[k-t], u[k-t-1], h[k-t])
        P[-t] = j
        b[-t] = bnext
        println("Level $t completed")
    end
    P,b
end

# update beliefs of neighbors j of factor a which is below variable i in the tree
function update_layer(H, X, variable_neigs, σ_gs::Vector{Int}, 
        efield::Vector{Tuple{T,T}}, bi::Tuple{T,T}, 
        ∂a::Vector{Int}, i::Int, a::Int, ukt, ukt1, hkt) where {T<:Real}
    uai_index = X[i,a]  # index of msg uai in the u vector (and of hia in the h vector)
    idx = findfirst(isequal(uai_index), ∂a)
    @assert idx!=nothing
    @assert uai_index != 0
    uai = ukt[uai_index]
    # @show msg2field(uai), uai_index
    # msg coming down from i to a
    hia = bi .- uai
    # @show msg2field(hia), msg2field(bi), msg2field(uai)
    # build vector h with the upcoming messages plus hia
    h = copy(hkt[∂a]); h[idx] = hia
    u = fill((zero(T),zero(T)), length(h))
    fMS_factor!(u, h)
    # make sure that maxsum is satisfied
    @assert u[idx] == ukt[uai_index]
    # @show msg2field.(u)
    ∂a_idx = msg2var.(∂a)
    # @show zip(∂a_idx, msg2field.(h))
    # println("Neighbors of factor $a are ", ∂a_idx)
    for c in eachindex(u)
        j = msg2var.(∂a[c])
        j == i && continue
        # find idx for the message to the other neighbor of j
        neigs_of_j = variable_neigs[j]
        idx_tmp = only(findall(!in(∂a), neigs_of_j))
        c_idx = neigs_of_j[idx_tmp]
        ucj = ukt1[c_idx]
        bj = normalize_max(ucj .+ efield[j] .+ u[c])
        bj_field = msg2field(bj)

        # println("\tVar $j: h_$j=", round(msg2field(efield[j]), digits=6),
        #      ".\tu_$(a)_to_$j = ", round(msg2field(u[c]), digits=6),
        #      ".\th_$(msg2fact(c_idx, H))_to_j = ", round(msg2field(ucj), digits=6),
        #      ".\tb_$j = ", round(bj_field, digits=6))

        bj_field == 0 && error("Variable $j is undecided when going from",
            " var $i to factor $a")
        (bj_field)*σ_gs[j] < 0 && return j, bj, msg2fact(c_idx, H)
    end
    error("Couldn't find a variable disagreing with the optimal solution.",
        " Going from variable $i to factor $a")
end

function fMS_factor!(u, h)
    for i in eachindex(u)
        u[i] = (0.0,-Inf)
        for j in eachindex(u)
            j==i && continue
            u[i] = msg_maxconv(u[i], h[j])
        end
        u[i] = normalize_max(u[i])    
    end
end

# function update_layer(H, X, σ_gs, bi::Tuple{T,T}, ∂a, i::Int, a::Int, ukt, ukt1, hkt) where {T<:Real}
#     uai_index = X[i,a]
#     @assert uai_index != 0
#     uai = ukt[uai_index]
#     m = bi .- uai
#     bj = (NaN, NaN)
#     for l in ∂a
#         l==uai_index && continue
#         u = (0.0,-Inf)
#         for r in eachindex(∂a)
#             (r==uai_index || r==l) && continue
#             u = msg_maxconv(u, hkt[r])
#         end
#         j = rowvals(X)[l]
#         u = u .+ m .+ efield[j]
#         if σ_gs[i]*(u[2]-u[1]) < 0
#             return j, u
#         end
#     end
#     error("Couldn't find a variable disagreing with the optimal solution")
# end