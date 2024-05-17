"""
    star_coloring(g::BipartiteGraph)

Compute a star coloring of the column vertices in the [`AdjacencyGraph`](@ref) `g` and return a vector of integer colors.

Def 4.5: A _star coloring_ is a distance-1 coloring such that every path on four vertices uses at least three colors.

The algorithm used is the greedy Algorithm 4.1.
"""
function star_coloring(g::AdjacencyGraph)
    n = length(columns(g))
    colors = zeros(Int, n)
    forbidden_colors = zeros(Int, n)
    for v in sort(columns(g); by=j -> length(neighbors(g, j)), rev=true)
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
