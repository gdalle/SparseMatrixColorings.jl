"""
    BipartiteGraph

Represent a bipartite graph between the rows and the columns of a non-symmetric `m × n` matrix `A`.

This graph is defined as `G = (R, C, E)` where `R = 1:m` is the set of row indices, `C = 1:n` is the set of column indices and `(i, j) ∈ E` whenever `A[i, j]` is nonzero.

# Fields

- `A_colmajor::AbstractMatrix`: output of [`col_major`](@ref) applied to `A` (useful to get neighbors of a column)
- `A_rowmajor::AbstractMatrix`: output of [`row_major`](@ref) applied to `A` (useful to get neighbors of a row)
"""
struct BipartiteGraph{M1<:AbstractMatrix,M2<:AbstractMatrix}
    A_colmajor::M1
    A_rowmajor::M2

    function BipartiteGraph(A::AbstractMatrix)
        A_colmajor = col_major(A)
        A_rowmajor = row_major(A)
        return new{typeof(A_colmajor),typeof(A_rowmajor)}(A_colmajor, A_rowmajor)
    end
end

rows(g::BipartiteGraph) = axes(g.A_colmajor, 1)
columns(g::BipartiteGraph) = axes(g.A_colmajor, 2)

neighbors_of_column(g::BipartiteGraph, j::Integer) = nz_in_col(g.A_colmajor, j)
neighbors_of_row(g::BipartiteGraph, i::Integer) = nz_in_row(g.A_rowmajor, i)
