"""
    AdjacencyGraph

Represent a graph between the columns of a symmetric `n × n` matrix `A` with nonzero diagonal elements.

This graph is defined as `G = (C, E)` where `C = 1:n` is the set of columns and `(i, j) ∈ E` whenever `A[i, j]` is nonzero for some `j ∈ 1:m`, `j ≠ i`.

# Fields

- `A_colmajor::AbstractMatrix`: output of [`col_major`](@ref) applied to `A`
"""
struct AdjacencyGraph{M<:AbstractMatrix}
    A_colmajor::M

    function AdjacencyGraph(A::AbstractMatrix)
        A_colmajor = col_major(A)
        return new{typeof(A_colmajor)}(A_colmajor)
    end
end

rows(g::AdjacencyGraph) = axes(g.A_colmajor, 1)
columns(g::AdjacencyGraph) = axes(g.A_colmajor, 2)

function neighbors(g::AdjacencyGraph, j::Integer)
    return filter(!isequal(j), nz_in_col(g.A_colmajor, j))
end
