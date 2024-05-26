"""
    GreedyColoringAlgorithm <: ADTypes.AbstractColoringAlgorithm

Greedy coloring algorithm for sparse Jacobians and Hessians, with configurable vertex order.

Compatible with the [ADTypes.jl coloring framework](https://sciml.github.io/ADTypes.jl/stable/#Coloring-algorithm).

# Constructor

    GreedyColoringAlgorithm(order::AbstractOrder=NaturalOrder())

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

"""
    column_coloring(A::AbstractMatrix, algo::GreedyColoringAlgorithm)

Compute a partial distance-2 coloring of the columns in the bipartite graph of the matrix `A`.

Function defined by ADTypes, re-exported by SparseMatrixColorings.

# Example

```jldoctest
using SparseMatrixColorings, SparseArrays

algo = GreedyColoringAlgorithm(SparseMatrixColorings.LargestFirst())

A = sparse([
    0 0 1 1 0
    1 0 0 0 1
    0 1 1 0 0
    0 1 1 0 1
])

column_coloring(A, algo)

# output

5-element Vector{Int64}:
 1
 2
 1
 2
 3
```
"""
function ADTypes.column_coloring(A::AbstractMatrix, algo::GreedyColoringAlgorithm)
    bg = bipartite_graph(A)
    return partial_distance2_coloring(bg, Val(2), algo.order)
end

"""
    row_coloring(A::AbstractMatrix, algo::GreedyColoringAlgorithm)

Compute a partial distance-2 coloring of the rows in the bipartite graph of the matrix `A`.

Function defined by ADTypes, re-exported by SparseMatrixColorings.

# Example

!!! warning
    Work in progress.
"""
function ADTypes.row_coloring(A::AbstractMatrix, algo::GreedyColoringAlgorithm)
    bg = bipartite_graph(A)
    return partial_distance2_coloring(bg, Val(1), algo.order)
end

"""
    symmetric_coloring(A::AbstractMatrix, algo::GreedyColoringAlgorithm)

Compute a star coloring of the columns in the adjacency graph of the symmetric matrix `A`.

Function defined by ADTypes, re-exported by SparseMatrixColorings.

# Example

!!! warning
    Work in progress.
"""
function ADTypes.symmetric_coloring(A::AbstractMatrix, algo::GreedyColoringAlgorithm)
    ag = adjacency_graph(A)
    return star_coloring(ag, algo.order)
end
