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

## Detailed interface

"""
    column_coloring_detailed(S::AbstractMatrix, algo::GreedyColoringAlgorithm)

Compute a partial distance-2 coloring of the columns in the bipartite graph of the matrix `S`, return an [`AbstractColoringResult`](@ref).

# Example

```jldoctest
using SparseMatrixColorings, SparseArrays

algo = GreedyColoringAlgorithm(SparseMatrixColorings.LargestFirst())

S = sparse([
    0 0 1 1 0
    1 0 0 0 1
    0 1 1 0 0
    0 1 1 0 1
])

column_colors(column_coloring_detailed(S, algo))

# output

5-element Vector{Int64}:
 1
 2
 1
 2
 3
```
"""
function column_coloring_detailed(S::AbstractMatrix, algo::GreedyColoringAlgorithm)
    bg = bipartite_graph(S)
    color = partial_distance2_coloring(bg, Val(2), algo.order)
    return SimpleColoringResult{:column,false}(S, color)
end

function column_coloring_detailed(S::SparseMatrixCSC, algo::GreedyColoringAlgorithm)
    bg = bipartite_graph(S)
    color = partial_distance2_coloring(bg, Val(2), algo.order)
    n = size(S, 1)
    I, J, _ = findnz(S)
    compressed_indices = zeros(Int, nnz(S))
    for k in eachindex(I, J, compressed_indices)
        i, j = I[k], J[k]
        c = color[j]
        # A[i, j] = B[i, c]
        compressed_indices[k] = (c - 1) * n + i
    end
    return SparseColoringResult{:column,false}(S, color, compressed_indices)
end

"""
    row_coloring_detailed(S::AbstractMatrix, algo::GreedyColoringAlgorithm)

Compute a partial distance-2 coloring of the rows in the bipartite graph of the matrix `S`, return an [`AbstractColoringResult`](@ref).

# Example

```jldoctest
using SparseMatrixColorings, SparseArrays

algo = GreedyColoringAlgorithm(SparseMatrixColorings.LargestFirst())

S = sparse([
    0 0 1 1 0
    1 0 0 0 1
    0 1 1 0 0
    0 1 1 0 1
])

row_colors(row_coloring_detailed(S, algo))

# output

4-element Vector{Int64}:
 2
 2
 3
 1
```
"""
function row_coloring_detailed(S::AbstractMatrix, algo::GreedyColoringAlgorithm)
    bg = bipartite_graph(S)
    color = partial_distance2_coloring(bg, Val(1), algo.order)
    return SimpleColoringResult{:row,false}(S, color)
end

function row_coloring_detailed(S::SparseMatrixCSC, algo::GreedyColoringAlgorithm)
    bg = bipartite_graph(S)
    color = partial_distance2_coloring(bg, Val(1), algo.order)
    n = size(S, 1)
    C = maximum(color)
    I, J, _ = findnz(S)
    compressed_indices = zeros(Int, nnz(S))
    for k in eachindex(I, J, compressed_indices)
        i, j = I[k], J[k]
        c = color[i]
        # A[i, j] = B[c, j]
        compressed_indices[k] = (j - 1) * C + c
    end
    return SparseColoringResult{:row,false}(S, color, compressed_indices)
end

"""
    symmetric_coloring(S::AbstractMatrix, algo::GreedyColoringAlgorithm)

Compute a star coloring of the columns in the adjacency graph of the symmetric matrix `S`, return an [`AbstractColoringResult`](@ref).

# Example

```jldoctest
using SparseMatrixColorings, SparseArrays

algo = GreedyColoringAlgorithm(SparseMatrixColorings.LargestFirst())

S = sparse([
    1 1 1 1
    1 1 0 0
    1 0 1 0
    1 0 0 1
])

column_colors(symmetric_coloring_detailed(S, algo))

# output

4-element Vector{Int64}:
 1
 2
 2
 2
```
"""
function symmetric_coloring_detailed(S::AbstractMatrix, algo::GreedyColoringAlgorithm)
    ag = adjacency_graph(S)
    color, star_set = star_coloring(ag, algo.order)
    # TODO: handle star_set
    return SimpleColoringResult{:column,true}(S, color)
end

function symmetric_coloring_detailed(S::SparseMatrixCSC, algo::GreedyColoringAlgorithm)
    ag = adjacency_graph(S)
    color, star_set = star_coloring(ag, algo.order)
    n = size(S, 1)
    I, J, _ = findnz(S)
    compressed_indices = zeros(Int, nnz(S))
    for k in eachindex(I, J, compressed_indices)
        i, j = I[k], J[k]
        l, c = symmetric_coefficient(i, j, color, star_set)
        # A[i, j] = B[l, c]
        compressed_indices[k] = (c - 1) * n + l
    end
    return SparseColoringResult{:column,true}(S, color, compressed_indices)
end

## ADTypes interface

"""
    ADTypes.column_coloring(S::AbstractMatrix, algo::GreedyColoringAlgorithm)

Call [`column_coloring_detailed`](@ref) and return only the vector of column colors.
"""
function ADTypes.column_coloring(S::AbstractMatrix, algo::GreedyColoringAlgorithm)
    return column_colors(column_coloring_detailed(S, algo))
end

"""
    ADTypes.row_coloring(S::AbstractMatrix, algo::GreedyColoringAlgorithm)

Call [`row_coloring_detailed`](@ref) and return only the vector of column colors.
"""
function ADTypes.row_coloring(S::AbstractMatrix, algo::GreedyColoringAlgorithm)
    return row_colors(row_coloring_detailed(S, algo))
end

"""
    ADTypes.symmetric_coloring(S::AbstractMatrix, algo::GreedyColoringAlgorithm)

Call [`symmetric_coloring_detailed`](@ref) and return only the vector of column colors.
"""
function ADTypes.symmetric_coloring(S::AbstractMatrix, algo::GreedyColoringAlgorithm)
    return column_colors(symmetric_coloring_detailed(S, algo))
end
