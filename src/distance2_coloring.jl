"""
    partial_distance2_coloring(g::Graph, S::AbstractVector{<:Integer})

Compute a distance-2 coloring of the vertex subset `S` in the [`Graph`](@ref) `g` and return a vector of integer colors, using the greedy Algorithm 3.2.

A _distance-2 coloring_ is such that two vertices have different colors if they are at distance at most 2.
"""
function partial_distance2_coloring(g::Graph, S::AbstractVector{<:Integer})
    color = zeros(Int, length(S))
    forbidden_colors = zeros(Int, length(S))
    for v in S
        for w in neighbors(g, v)
            for x in neighbors(g, w)
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
