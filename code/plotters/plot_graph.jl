using LightGraphs, GraphRecipes, Plots

# Returns the proper square adjacency matrix
function full_adjmat(H::SparseMatrixCSC)
    m,n = size(H)
    A = [zeros(Int,m,m) H;
         permutedims(H) zeros(Int,n,n)]
    return A
end

function Plots.plot(H::SparseMatrixCSC; varnames=1:size(H,2), 
    source::BitVector=falses(size(H,2)), factnames=1:size(H,1), method=:spring,
    randseed::Int=abs(rand(Int)), plt_kw...)
    
    # Plots.pyplot()
    # Plots.gr()
    m,n = size(H)
    g = SimpleGraph(full_adjmat(H))
    if ne(g) == 0
        println("Graph contains no edges")
        return nothing
    end
    nodenames = [""*string(i)*"" for i in [factnames; varnames]]
    node_idx = [ones(Int,m); 2*ones(Int,n)]
    shapes = [:rect, :circle]
    nodeshape = shapes[node_idx]
    cols = [:white, :yellow]
    nodecolor = cols[vcat(ones(Int,m), source.+1)]
        
    Random.seed!(randseed)  # control random layout changes
    return graphplot(g, curves=false, names=nodenames, 
        markercolor=nodecolor,
        nodeshape = nodeshape, method=method, nodesize=0.15, fontsize=7;
        plt_kw...)
end

# function Plots.plot(H::SparseMatrixCSC; varnames=1:size(H,2), factnames=1:size(H,1),
#     highlighted_nodes=Int[], highlighted_factors=Int[], 
#     highlighted_edges::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[], method=:spring,
#     randseed::Int=abs(rand(Int)), plt_kw...)
    
#     Plots.pyplot()
#     m,n = size(H)
#     if typeof(highlighted_nodes)==Int
#         highlighted_nodes = [highlighted_nodes]
#     end
#     g = SimpleGraph(full_adjmat(H))
#     if ne(g) == 0
#         println("Graph contains no edges")
#         return nothing
#     end
#     nodenames = [""*string(i)*"" for i in [factnames; varnames]]
#     node_idx = [ones(Int,m); 2*ones(Int,n)]
#     node_idx[highlighted_factors] .= 3
#     node_idx[m .+ highlighted_nodes] .= 4
#     shapes = [:rect, :circle, :rect, :circle]
#     nodeshape = shapes[node_idx]
#     colors = [:white, :yellow, :red, :orange]
#     nodecolor = colors[node_idx]
#     strokewidths = [0.5, 0.1, 0.5, 0.1]
#     nodestrokewidth = strokewidths[node_idx]
#     edges_idx = [1 for _ in edges(g)]
#     edgecolor = [a==1 ? :black : :none for a in adjacency_matrix(g)]
#     highlighted_edges_ = [(t[1],t[2]+m) for t in highlighted_edges]
#     edgecolor[CartesianIndex.(highlighted_edges_)] .= :red
    
#     Random.seed!(randseed)  # control random layout changes
#     return graphplot(g, curves=false, names=nodenames,
#         nodeshape = nodeshape, nodecolor=colors[node_idx],
#         method=method, nodesize=0.15, fontsize=7, 
#         nodestrokewidth=nodestrokewidth, edgecolor=edgecolor; plt_kw...)
# end