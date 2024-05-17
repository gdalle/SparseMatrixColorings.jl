"""
    GreedyColoringAlgorithm <: ADTypes.AbstractColoringAlgorithm

Matrix coloring algorithm for sparse Jacobians and Hessians.

Compatible with the [ADTypes.jl coloring framework](https://sciml.github.io/ADTypes.jl/stable/#Coloring-algorithm).

# Implements

- `ADTypes.column_coloring` with a partial distance-2 coloring of the bipartite graph
- `ADTypes.row_coloring` with a partial distance-2 coloring of the bipartite graph
- `ADTypes.symmetric_coloring` with a star coloring of the adjacency graph
"""
struct GreedyColoringAlgorithm <: ADTypes.AbstractColoringAlgorithm end

function ADTypes.column_coloring(A::AbstractMatrix, ::GreedyColoringAlgorithm)
    g = BipartiteGraph(A)
    return distance2_column_coloring(g)
end

function ADTypes.row_coloring(A::AbstractMatrix, ::GreedyColoringAlgorithm)
    g = BipartiteGraph(A)
    return distance2_row_coloring(g)
end

function ADTypes.symmetric_coloring(A::AbstractMatrix, ::GreedyColoringAlgorithm)
    g = AdjacencyGraph(A)
    return star_coloring(g)
end
