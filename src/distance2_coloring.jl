"""
    distance2_column_coloring(g::BipartiteGraph)

Compute a distance-2 coloring of the column vertices in the [`BipartiteGraph`](@ref) `g` and return a vector of integer colors.

A _distance-2 coloring_ is such that two vertices have different colors if they are at distance at most 2.

The algorithm used is the greedy Algorithm 3.2.
"""
function distance2_column_coloring(g::BipartiteGraph)
    n = length(columns(g))
    colors = zeros(Int, n)
    forbidden_colors = zeros(Int, n)
    for v in sort(columns(g); by=j -> length(neighbors_of_column(g, j)), rev=true)
        for w in neighbors_of_column(g, v)
            for x in neighbors_of_row(g, w)
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
    distance2_row_coloring(g::BipartiteGraph)

Compute a distance-2 coloring of the row vertices in the [`BipartiteGraph`](@ref) `g` and return a vector of integer colors.

A _distance-2 coloring_ is such that two vertices have different colors if they are at distance at most 2.

The algorithm used is the greedy Algorithm 3.2.
"""
function distance2_row_coloring(g::BipartiteGraph)
    m = length(rows(g))
    colors = zeros(Int, m)
    forbidden_colors = zeros(Int, m)
    for v in sort(rows(g); by=i -> length(neighbors_of_row(g, i)), rev=true)
        for w in neighbors_of_row(g, v)
            for x in neighbors_of_column(g, w)
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
