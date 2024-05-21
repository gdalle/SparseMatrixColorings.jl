"""
    partial_distance2_coloring(bg::BipartiteGraph, ::Val{side}, order::AbstractOrder)

Compute a distance-2 coloring of the given `side` (`1` or `2`) in the bipartite graph `bg` and return a vector of integer colors.

A _distance-2 coloring_ is such that two vertices have different colors if they are at distance at most 2.

The vertices are colored in a greedy fashion, following the `order` supplied.

# See also

- [`BipartiteGraph`](@ref)
- [`AbstractOrder`](@ref)
"""
function partial_distance2_coloring(
    bg::BipartiteGraph, ::Val{side}, order::AbstractOrder
) where {side}
    other_side = 3 - side
    colors = zeros(Int, length(bg, Val(side)))
    forbidden_colors = zeros(Int, length(bg, Val(side)))
    for v in vertices(bg, Val(side), order)
        for w in neighbors(bg, Val(side), v)
            for x in neighbors(bg, Val(other_side), w)
                if !iszero(colors[x])
                    forbidden_colors[colors[x]] = v
                end
            end
        end
        for c in eachindex(forbidden_colors)
            if forbidden_colors[c] != v
                colors[v] = c
                break
            end
        end
    end
    return colors
end

"""
    star_coloring1(g::Graph, order::AbstractOrder)

Compute a star coloring of all vertices in the adjacency graph `g` and return a vector of integer colors.

A _star coloring_ is a distance-1 coloring such that every path on 4 vertices uses at least 3 colors.

The vertices are colored in a greedy fashion, following the `order` supplied.

# See also

- [`Graph`](@ref)
- [`AbstractOrder`](@ref)
"""
function star_coloring1(g::Graph, order::AbstractOrder)
    n = length(g)
    colors = zeros(Int, n)
    forbidden_colors = zeros(Int, n)
    for v in vertices(g, order)
        for w in neighbors(g, v)
            if !iszero(colors[w])  # w is colored
                forbidden_colors[colors[w]] = v
            end
            for x in neighbors(g, w)
                if !iszero(colors[x]) && iszero(colors[w])  # w is not colored
                    forbidden_colors[colors[x]] = v
                else
                    for y in neighbors(g, x)
                        if !iszero(colors[y]) && y != w
                            if colors[y] == colors[w]
                                forbidden_colors[colors[x]] = v
                                break
                            end
                        end
                    end
                end
            end
        end
        for c in eachindex(forbidden_colors)
            if forbidden_colors[c] != v
                colors[v] = c
                break
            end
        end
    end
    return colors
end
