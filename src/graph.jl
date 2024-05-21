struct Graph{T<:Integer}
    colptr::Vector{T}
    rowval::Vector{T}
end

Graph(A::SparseMatrixCSC) = Graph(A.colptr, A.rowval)

Base.length(g::Graph) = length(g.colptr) - 1

neighbors(g::Graph, v::Integer) = view(g.rowval, g.colptr[v]:(g.colptr[v + 1] - 1))

## Adjacency graph

"""
    AdjacencyGraph

Undirected graph representing the nonzeros of a symmetrix matrix (typically a Hessian matrix).

The adjacency graph of a symmetrix matrix `A ∈ ℝ^{n × n}` is `G(A) = (V, E)` where

- `V = 1:n` is the set of rows or columns
- `(i, j) ∈ E` whenever `A[i, j] ≠ 0` and `i ≠ j`
"""
struct AdjacencyGraph{T}
    g::Graph{T}
end

function AdjacencyGraph(H::SparseMatrixCSC)
    g = Graph(H - Diagonal(H))
    return AdjacencyGraph(g)
end

Base.length(ag::AdjacencyGraph) = length(ag.g)

"""
    neighbors(ag::AdjacencyGraph, v::Integer)

Return the neighbors of `v` in the graph `ag`.
"""
neighbors(ag::AdjacencyGraph, v::Integer) = neighbors(ag.g, v)

degree(ag::AdjacencyGraph, v::Integer) = length(neighbors(ag, v))

## Bipartite graph

"""
    BipartiteGraph

Undirected bipartite graph representing the nonzeros of a non-symmetric matrix (typically a Jacobian matrix).

The bipartite graph of a matrix `A ∈ ℝ^{m × n}` is `Gb(A) = (V₁, V₂, E)` where

- `V₁ = 1:m` is the set of rows `i`
- `V₂ = 1:n` is the set of columns `j`
- `(i, j) ∈ E` whenever `A[i, j] ≠ 0`
"""
struct BipartiteGraph{T}
    g1::Graph{T}
    g2::Graph{T}
end

function BipartiteGraph(J::SparseMatrixCSC)
    g1 = Graph(SparseMatrixCSC(transpose(J)))  # rows to columns
    g2 = Graph(J)  # columns to rows
    return BipartiteGraph(g1, g2)
end

Base.length(bg::BipartiteGraph, ::Val{1}) = length(bg.g1)
Base.length(bg::BipartiteGraph, ::Val{2}) = length(bg.g2)

"""
    neighbors(bg::BipartiteGraph, Val(side), v::Integer)

Return the neighbors of `v`, which is a vertex from the specified `side` (`1` or `2`), in the graph `bg`.
"""
neighbors(bg::BipartiteGraph, ::Val{1}, v::Integer) = neighbors(bg.g1, v)
neighbors(bg::BipartiteGraph, ::Val{2}, v::Integer) = neighbors(bg.g2, v)

function degree(bg::BipartiteGraph, ::Val{side}, v::Integer) where {side}
    return length(neighbors(bg, Val(side), v))
end
