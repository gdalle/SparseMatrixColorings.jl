## Standard graph

"""
    Graph{T}

Undirected graph structure stored in Compressed Sparse Column (CSC) format.

# Fields

- `colptr::Vector{T}`: same as for `SparseMatrixCSC`
- `rowval::Vector{T}`: same as for `SparseMatrixCSC`
"""
struct Graph{T<:Integer}
    colptr::Vector{T}
    rowval::Vector{T}
end

Graph(A::SparseMatrixCSC) = Graph(A.colptr, A.rowval)
Graph(A::AbstractMatrix) = Graph(sparse(A))

Base.length(g::Graph) = length(g.colptr) - 1
SparseArrays.nnz(g::Graph) = length(g.rowval)

vertices(g::Graph) = 1:length(g)
neighbors(g::Graph, v::Integer) = view(g.rowval, g.colptr[v]:(g.colptr[v + 1] - 1))
degree(g::Graph, v::Integer) = length(g.colptr[v]:(g.colptr[v + 1] - 1))

maximum_degree(g::Graph) = maximum(Base.Fix1(degree, g), vertices(g))
minimum_degree(g::Graph) = minimum(Base.Fix1(degree, g), vertices(g))

## Bipartite graph

"""
    BipartiteGraph{T}

Undirected bipartite graph structure stored in bidirectional Compressed Sparse Column format (redundancy allows for faster access).

A bipartite graph has two "sides", which we number `1` and `2`.

# Fields

- `g1::Graph{T}`: contains the neighbors for vertices on side `1`
- `g2::Graph{T}`: contains the neighbors for vertices on side `2`
"""
struct BipartiteGraph{T<:Integer}
    g1::Graph{T}
    g2::Graph{T}
end

Base.length(bg::BipartiteGraph, ::Val{1}) = length(bg.g1)
Base.length(bg::BipartiteGraph, ::Val{2}) = length(bg.g2)
SparseArrays.nnz(bg::BipartiteGraph) = nnz(bg.g1)

"""
    vertices(bg::BipartiteGraph, Val(side))

Return the list of vertices of `bg` from the specified `side` as a range `1:n`.
"""
vertices(bg::BipartiteGraph, ::Val{side}) where {side} = 1:length(bg, Val(side))

"""
    neighbors(bg::BipartiteGraph, Val(side), v::Integer)

Return the neighbors of `v` (a vertex from the specified `side`, `1` or `2`), in the graph `bg`.
"""
neighbors(bg::BipartiteGraph, ::Val{1}, v::Integer) = neighbors(bg.g1, v)
neighbors(bg::BipartiteGraph, ::Val{2}, v::Integer) = neighbors(bg.g2, v)

degree(bg::BipartiteGraph, ::Val{1}, v::Integer) = degree(bg.g1, v)
degree(bg::BipartiteGraph, ::Val{2}, v::Integer) = degree(bg.g2, v)

function maximum_degree(bg::BipartiteGraph, ::Val{side}) where {side}
    return maximum(v -> degree(bg, Val(side), v), vertices(bg, Val(side)))
end

function minimum_degree(bg::BipartiteGraph, ::Val{side}) where {side}
    return minimum(v -> degree(bg, Val(side), v), vertices(bg, Val(side)))
end

## Construct from matrices

"""
    adjacency_graph(H::AbstractMatrix)

Return a [`Graph`](@ref) representing the nonzeros of a symmetric matrix (typically a Hessian matrix).

The adjacency graph of a symmetrix matric `A ∈ ℝ^{n × n}` is `G(A) = (V, E)` where

- `V = 1:n` is the set of rows or columns `i`/`j`
- `(i, j) ∈ E` whenever `A[i, j] ≠ 0` and `i ≠ j`
"""
adjacency_graph(H::SparseMatrixCSC) = Graph(H - Diagonal(H))
adjacency_graph(H::AbstractMatrix) = adjacency_graph(sparse(H))

"""
    bipartite_graph(J::AbstractMatrix)

Return a [`BipartiteGraph`](@ref) representing the nonzeros of a non-symmetric matrix (typically a Jacobian matrix).

The bipartite graph of a matrix `A ∈ ℝ^{m × n}` is `Gb(A) = (V₁, V₂, E)` where

- `V₁ = 1:m` is the set of rows `i`
- `V₂ = 1:n` is the set of columns `j`
- `(i, j) ∈ E` whenever `A[i, j] ≠ 0`
"""
function bipartite_graph(J::SparseMatrixCSC)
    g1 = Graph(transpose(J))  # rows to columns
    g2 = Graph(J)  # columns to rows
    return BipartiteGraph(g1, g2)
end

bipartite_graph(J::AbstractMatrix) = bipartite_graph(sparse(J))

"""
    column_intersection_graph(J::AbstractMatrix)

Return a [`Graph`](@ref) representing the column intersections of a non-symmetric matrix (typically a Jacobian matrix).

The column intersection graph of a matrix `A ∈ ℝ^{m × n}` is `Gc(A) = (V, E)` where

- `V = 1:n` is the set of columns `j`
- `(j1, j2) ∈ E` whenever `A[:, j1] ∩ A[:, j2] ≠ ∅`
"""
function column_intersection_graph(J::SparseMatrixCSC)
    A = transpose(J) * J
    return adjacency_graph(A - Diagonal(A))
end

column_intersection_graph(J::AbstractMatrix) = column_intersection_graph(sparse(J))
