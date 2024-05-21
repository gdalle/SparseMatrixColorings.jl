"""
    GreedyColoringAlgorithm <: ADTypes.AbstractColoringAlgorithm

Matrix coloring algorithm for sparse Jacobians and Hessians.

Compatible with the [ADTypes.jl coloring framework](https://sciml.github.io/ADTypes.jl/stable/#Coloring-algorithm).

# Constructor

    GreedyColoringAlgorithm(order::AbstractOrder=NaturalOrder())

# Implements

- [`ADTypes.column_coloring`](@extref ADTypes) and [`ADTypes.row_coloring`](@extref ADTypes) with a partial distance-2 coloring of the bipartite graph
- [`ADTypes.symmetric_coloring`](@extref ADTypes) with a star coloring of the adjacency graph
"""
struct GreedyColoringAlgorithm{O<:AbstractOrder} <: ADTypes.AbstractColoringAlgorithm
    order::O
end

GreedyColoringAlgorithm() = GreedyColoringAlgorithm(NaturalOrder())

function Base.show(io::IO, algo::GreedyColoringAlgorithm)
    return print(io, "GreedyColoringAlgorithm($(algo.order))")
end

function ADTypes.column_coloring(A::AbstractMatrix, algo::GreedyColoringAlgorithm)
    bg = bipartite_graph(A)
    return partial_distance2_coloring(bg, Val(2), algo.order)
end

function ADTypes.row_coloring(A::AbstractMatrix, algo::GreedyColoringAlgorithm)
    bg = bipartite_graph(A)
    return partial_distance2_coloring(bg, Val(1), algo.order)
end

function ADTypes.symmetric_coloring(A::AbstractMatrix, algo::GreedyColoringAlgorithm)
    ag = adjacency_graph(A)
    return star_coloring1(ag, algo.order)
end
