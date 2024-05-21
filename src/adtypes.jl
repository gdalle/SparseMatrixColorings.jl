"""
    GreedyColoringAlgorithm <: ADTypes.AbstractColoringAlgorithm

Greedy coloring algorithm for sparse Jacobians and Hessians, with configurable vertex order.

Compatible with the [ADTypes.jl coloring framework](https://sciml.github.io/ADTypes.jl/stable/#Coloring-algorithm).

# Constructor

    GreedyColoringAlgorithm(order::AbstractOrder=NaturalOrder())

# Implements

- [`ADTypes.column_coloring`](@extref ADTypes) and [`ADTypes.row_coloring`](@extref ADTypes) with a partial distance-2 coloring of the bipartite graph
- [`ADTypes.symmetric_coloring`](@extref ADTypes) with a star coloring of the adjacency graph

# Example use

```jldoctest
using ADTypes, SparseMatrixColorings, SparseArrays

algo = GreedyColoringAlgorithm(SparseMatrixColorings.LargestFirst())
A = sparse([
    0 0 1 1 0
    1 0 0 0 1
    0 1 1 0 0
    0 1 1 0 1
])
ADTypes.column_coloring(A, algo)

# output

5-element Vector{Int64}:
 1
 2
 1
 2
 3
```

# See also

- [`AbstractOrder`](@ref)
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
