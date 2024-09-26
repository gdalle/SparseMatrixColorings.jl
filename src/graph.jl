## Standard graph

"""
    Graph{loops,T}

Undirected graph structure stored in Compressed Sparse Column (CSC) format.
Note that the underyling CSC may not be square, so the term "graph" is slightly abusive.

The type parameter `loops` must be set to:
- `true` if coefficients `(i, i)` present in the CSC are counted as edges in the graph (e.g. for each half of a bipartite graph)
- `false` otherwise (e.g. for an adjacency graph)

# Fields

- `colptr::Vector{T}`: same as for `SparseMatrixCSC`
- `rowval::Vector{T}`: same as for `SparseMatrixCSC`
"""
struct Graph{loops,T<:Integer}
    colptr::Vector{T}
    rowval::Vector{T}
end

function Graph{loops}(S::SparseMatrixCSC{Tv,Ti}) where {loops,Tv,Ti}
    return Graph{loops,Ti}(S.colptr, S.rowval)
end

Base.length(g::Graph) = length(g.colptr) - 1

SparseArrays.nnz(g::Graph{true}) = length(g.rowval)

function SparseArrays.nnz(g::Graph{false})
    n = 0
    for j in 1:(length(g.colptr) - 1)
        for k in g.colptr[j]:(g.colptr[j + 1] - 1)
            i = g.rowval[k]
            if i != j
                n += 1
            end
        end
    end
    return n
end

vertices(g::Graph) = 1:length(g)

function neighbors(g::Graph{true}, v::Integer)
    return view(g.rowval, g.colptr[v]:(g.colptr[v + 1] - 1))
end

function neighbors(g::Graph{false}, v::Integer)
    neighbors_with_loops = view(g.rowval, g.colptr[v]:(g.colptr[v + 1] - 1))
    return Iterators.filter(!=(v), neighbors_with_loops)  # TODO: optimize
end

function degree(g::Graph{true}, v::Integer)
    return length(g.colptr[v]:(g.colptr[v + 1] - 1))
end

function degree(g::Graph{false}, v::Integer)
    d = length(g.colptr[v]:(g.colptr[v + 1] - 1))
    for k in g.colptr[v]:(g.colptr[v + 1] - 1)
        if g.rowval[k] == v
            d -= 1
        end
    end
    return d
end

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
    g1::Graph{true,T}
    g2::Graph{true,T}
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
    adjacency_graph(A::SparseMatrixCSC)

Return a [`Graph`](@ref) representing the nonzeros of a symmetric matrix (typically a Hessian matrix).

The adjacency graph of a symmetrix matric `A ∈ ℝ^{n × n}` is `G(A) = (V, E)` where

- `V = 1:n` is the set of rows or columns `i`/`j`
- `(i, j) ∈ E` whenever `A[i, j] ≠ 0` and `i ≠ j`

# References

> [_What Color Is Your Jacobian? Graph Coloring for Computing Derivatives_](https://epubs.siam.org/doi/10.1137/S0036144504444711), Gebremedhin et al. (2005)
"""
adjacency_graph(A::SparseMatrixCSC) = Graph{false}(A)

"""
    bipartite_graph(A::SparseMatrixCSC; symmetric_pattern::Bool)

Return a [`BipartiteGraph`](@ref) representing the nonzeros of a non-symmetric matrix (typically a Jacobian matrix).

The bipartite graph of a matrix `A ∈ ℝ^{m × n}` is `Gb(A) = (V₁, V₂, E)` where

- `V₁ = 1:m` is the set of rows `i`
- `V₂ = 1:n` is the set of columns `j`
- `(i, j) ∈ E` whenever `A[i, j] ≠ 0`

When `symmetric_pattern` is `true`, this construction is more efficient.

# References

> [_What Color Is Your Jacobian? Graph Coloring for Computing Derivatives_](https://epubs.siam.org/doi/10.1137/S0036144504444711), Gebremedhin et al. (2005)
"""
function bipartite_graph(A::SparseMatrixCSC; symmetric_pattern::Bool=false)
    g2 = Graph{true}(A)  # columns to rows
    if symmetric_pattern
        checksquare(A)  # proxy for checking full symmetry
        g1 = g2
    else
        g1 = Graph{true}(convert(SparseMatrixCSC, transpose(A)))  # rows to columns
    end
    return BipartiteGraph(g1, g2)
end
