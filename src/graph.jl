## Standard graph

"""
    SparsityPatternCSC{Ti<:Integer}

Store a sparse matrix (in CSC) without its values, keeping only the pattern of nonzeros.

# Fields

Copied from `SparseMatrixCSC`:

- `m::Int`: number of rows
- `n::Int`: number of columns
- `colptr::Vector{Ti}`: column `j` is in `colptr[j]:(colptr[j+1]-1)`
- `rowval::Vector{Ti}`: row indices of stored values
"""
struct SparsityPatternCSC{Ti<:Integer} <: AbstractMatrix{Bool}
    m::Int
    n::Int
    colptr::Vector{Ti}
    rowval::Vector{Ti}
end

SparsityPatternCSC(A::SparseMatrixCSC) = SparsityPatternCSC(A.m, A.n, A.colptr, A.rowval)

Base.eltype(::SparsityPatternCSC{T}) where {T} = T
Base.size(S::SparsityPatternCSC) = (S.m, S.n)
Base.size(S::SparsityPatternCSC, d::Integer) = d::Integer <= 2 ? size(S)[d] : 1
Base.axes(S::SparsityPatternCSC, d::Integer) = Base.OneTo(size(S, d))

SparseArrays.nnz(S::SparsityPatternCSC) = length(S.rowval)
SparseArrays.rowvals(S::SparsityPatternCSC) = S.rowval
SparseArrays.nzrange(S::SparsityPatternCSC, j::Integer) = S.colptr[j]:(S.colptr[j + 1] - 1)

"""
    transpose(S::SparsityPatternCSC)

Return a [`SparsityPatternCSC`](@ref) corresponding to the transpose of `S`.
"""
function Base.transpose(S::SparsityPatternCSC{T}) where {T}
    m, n = size(S)
    nnzA = nnz(S)
    A_colptr = S.colptr
    A_rowval = S.rowval

    # Allocate storage for the column pointers and row indices of B = Aᵀ
    B_colptr = zeros(T, m + 1)
    B_rowval = Vector{T}(undef, nnzA)

    # Count the number of non-zeros for each row of A.
    # It corresponds to the number of non-zeros for each column of B = Aᵀ.
    for k in 1:nnzA
        i = A_rowval[k]
        B_colptr[i] += 1
    end

    # Compute the cumulative sum to determine the starting positions of rows in B_rowval
    counter = 1
    for col in 1:m
        nnz_col = B_colptr[col]
        B_colptr[col] = counter
        counter += nnz_col
    end
    B_colptr[m + 1] = counter

    # Store the row indices for each column of B = Aᵀ
    for j in 1:n
        for index in A_colptr[j]:(A_colptr[j + 1] - 1)
            i = A_rowval[index]

            # Update B_rowval for the non-zero B[j,i].
            # It corresponds to the non-zero A[i,j].
            pos = B_colptr[i]
            B_rowval[pos] = j
            B_colptr[i] += 1
        end
    end

    # Fix offsets of B_colptr to restore correct starting positions
    for col in m:-1:2
        B_colptr[col] = B_colptr[col - 1]
    end
    B_colptr[1] = 1

    return SparsityPatternCSC{T}(n, m, B_colptr, B_rowval)
end

# copied from SparseArrays.jl
function Base.getindex(S::SparsityPatternCSC, i0::Integer, i1::Integer)
    r1 = Int(S.colptr[i1])
    r2 = Int(S.colptr[i1 + 1] - 1)
    (r1 > r2) && return false
    r1 = searchsortedfirst(rowvals(S), i0, r1, r2, Base.Order.Forward)
    return ((r1 > r2) || (rowvals(S)[r1] != i0)) ? false : true
end

"""
    bidirectional_pattern(A::AbstractMatrix; symmetric_pattern::Bool)

Return a [`SparsityPatternCSC`](@ref) corresponding to the matrix `[0 Aᵀ; A 0]`, with a minimum of allocations.
"""
function bidirectional_pattern(A::AbstractMatrix; symmetric_pattern::Bool)
    return bidirectional_pattern(SparsityPatternCSC(SparseMatrixCSC(A)); symmetric_pattern)
end

function bidirectional_pattern(S::SparsityPatternCSC{T}; symmetric_pattern::Bool) where {T}
    m, n = size(S)
    p = m + n
    nnzS = nnz(S)
    rowval = Vector{T}(undef, 2 * nnzS)
    colptr = zeros(T, p + 1)
    edge_to_index = Vector{T}(undef, 2 * nnzS)

    # Update rowval and colptr for the block A
    for i in 1:nnzS
        rowval[i] = S.rowval[i] + n
        edge_to_index[i] = i
    end
    for j in 1:n
        colptr[j] = S.colptr[j]
    end

    # Update rowval and colptr for the block Aᵀ
    if symmetric_pattern
        # Use colptr[n+1:n+m] to store the offsets during the update of edge_to_index
        offsets = colptr

        # We use the sparsity pattern of A for Aᵀ
        for k in 1:nnzS
            r = S.rowval[k]
            rowval[nnzS + k] = r
            pos = S.colptr[r] + offsets[n + r]
            edge_to_index[nnzS + pos] = edge_to_index[k]
            offsets[n + r] += 1
        end
        # m and n are identical because symmetric_pattern is true
        for j in 1:m
            colptr[n + j] = nnzS + S.colptr[j]
        end
        colptr[p + 1] = 2 * nnzS + 1
    else
        # We need to determine the sparsity pattern of Aᵀ
        # We adapt the code of transpose(SparsityPatternCSC) in graph.jl
        for k in 1:nnzS
            i = S.rowval[k]
            colptr[n + i] += 1
        end

        counter = 1
        for col in (n + 1):p
            nnz_col = colptr[col]
            colptr[col] = nnzS + counter
            counter += nnz_col
        end

        for j in 1:n
            for index in S.colptr[j]:(S.colptr[j + 1] - 1)
                i = S.rowval[index]
                pos = colptr[n + i]
                rowval[pos] = j
                edge_to_index[pos] = edge_to_index[index]
                colptr[n + i] += 1
            end
        end

        colptr[p + 1] = nnzS + counter
        for col in p:-1:(n + 2)
            colptr[col] = colptr[col - 1]
        end
        colptr[n + 1] = nnzS + 1
    end

    # Create the SparsityPatternCSC of the augmented adjacency matrix
    S_and_Sᵀ = SparsityPatternCSC{T}(p, p, colptr, rowval)
    return S_and_Sᵀ, edge_to_index
end

function build_edge_to_index(S::SparsityPatternCSC{T}) where {T}
    # edge_to_index gives an index for each edge
    edge_to_index = Vector{T}(undef, nnz(S))
    offsets = zeros(T, S.n)
    nb_self_loops = 0
    counter = 0
    rvS = rowvals(S)
    for j in axes(S, 2)
        for k in nzrange(S, j)
            i = rvS[k]
            if i > j
                counter += 1
                edge_to_index[k] = counter
                k2 = S.colptr[i] + offsets[i]
                edge_to_index[k2] = counter
                offsets[i] += 1
            elseif i == j
                # this should never be used, make sure it errors
                edge_to_index[k] = 0
                nb_self_loops += 1
            end
        end
    end
    return edge_to_index, nb_self_loops
end

## Adjacency graph

"""
    AdjacencyGraph{T,augmented_graph}

Undirected graph without self-loops representing the nonzeros of a symmetric matrix (typically a Hessian matrix).

The adjacency graph of a symmetric matrix `A ∈ ℝ^{n × n}` is `G(A) = (V, E)` where

- `V = 1:n` is the set of rows or columns `i`/`j`
- `(i, j) ∈ E` whenever `A[i, j] ≠ 0` and `i ≠ j`

# Constructors

    AdjacencyGraph(A::SparseMatrixCSC; augmented_graph::Bool=false)

# Fields

- `S::SparsityPatternCSC{T}`: Underlying sparsity pattern, which represents an augmented graph whenever `augmented_graph` is `true`. Here, "augmented graph" means the sparsity pattern of the augmented matrix `H = [0 Jᵀ; J 0]`.
- `edge_to_index::Vector{T}`: A vector mapping each nonzero of `S` to a unique edge index (ignoring diagonal and accounting for symmetry, so that `(i, j)` and `(j, i)` get the same index)

# References

> [_What Color Is Your Jacobian? SparsityPatternCSC Coloring for Computing Derivatives_](https://epubs.siam.org/doi/10.1137/S0036144504444711), Gebremedhin et al. (2005)
"""
struct AdjacencyGraph{T<:Integer,augmented_graph}
    S::SparsityPatternCSC{T}
    edge_to_index::Vector{T}
    nb_self_loops::Int
    original_size::Tuple{Int,Int}
end

Base.eltype(::AdjacencyGraph{T}) where {T} = T

function AdjacencyGraph(
    S::SparsityPatternCSC{T},
    edge_to_index::Vector{T},
    nb_self_loops::Int;
    augmented_graph::Bool=false,
    original_size::Tuple{Int,Int}=size(S),
) where {T}
    return AdjacencyGraph{T,augmented_graph}(S, edge_to_index, nb_self_loops, original_size)
end

function AdjacencyGraph(
    S::SparsityPatternCSC;
    augmented_graph::Bool=false,
    original_size::Tuple{Int,Int}=size(S),
)
    edge_to_index, nb_self_loops = build_edge_to_index(S)
    return AdjacencyGraph(S, edge_to_index, nb_self_loops; augmented_graph, original_size)
end

function AdjacencyGraph(
    A::SparseMatrixCSC; augmented_graph::Bool=false, original_size::Tuple{Int,Int}=size(A)
)
    return AdjacencyGraph(SparsityPatternCSC(A); augmented_graph, original_size)
end

function AdjacencyGraph(
    A::AbstractMatrix; augmented_graph::Bool=false, original_size::Tuple{Int,Int}=size(A)
)
    return AdjacencyGraph(SparseMatrixCSC(A); augmented_graph, original_size)
end

pattern(g::AdjacencyGraph) = g.S
edge_indices(g::AdjacencyGraph) = g.edge_to_index
nb_vertices(g::AdjacencyGraph) = pattern(g).n
vertices(g::AdjacencyGraph) = 1:nb_vertices(g)
augmented_graph(::AdjacencyGraph{T,ag}) where {T,ag} = ag

function neighbors(g::AdjacencyGraph, v::Integer)
    S = pattern(g)
    neighbors_v = view(rowvals(S), nzrange(S, v))
    return neighbors_v
end

function neighbors_with_edge_indices(g::AdjacencyGraph, v::Integer)
    S = pattern(g)
    range_v = nzrange(S, v)
    neighbors_v = view(rowvals(S), range_v)
    edges_indices_v = view(edge_indices(g), range_v)
    return zip(neighbors_v, edges_indices_v)
end

degree(g::AdjacencyGraph{T,true}, v::Integer) where {T} = g.S.colptr[v + 1] - g.S.colptr[v]

function degree(g::AdjacencyGraph{T,false}, v::Integer) where {T}
    neigh = neighbors(g, v)
    has_selfloop = insorted(v, neigh)
    return g.S.colptr[v + 1] - g.S.colptr[v] - has_selfloop
end

nb_edges(g::AdjacencyGraph) = (nnz(g.S) - g.nb_self_loops) ÷ 2

maximum_degree(g::AdjacencyGraph) = maximum(Base.Fix1(degree, g), vertices(g))
minimum_degree(g::AdjacencyGraph) = minimum(Base.Fix1(degree, g), vertices(g))

function has_neighbor(g::AdjacencyGraph, v::Integer, u::Integer)
    for w in neighbors(g, v)
        augmented_graph(g) || (v == w && continue)
        if w == u
            return true
        end
    end
    return false
end

function degree_in_subset(g::AdjacencyGraph, v::Integer, subset::AbstractVector{<:Integer})
    d = 0
    for u in subset
        if has_neighbor(g, v, u)
            d += 1
        end
    end
    return d
end

## Bipartite graph

"""
    BipartiteGraph{T}

Undirected bipartite graph representing the nonzeros of a non-symmetric matrix (typically a Jacobian matrix).

The bipartite graph of a matrix `A ∈ ℝ^{m × n}` is `Gb(A) = (V₁, V₂, E)` where

- `V₁ = 1:m` is the set of rows `i`
- `V₂ = 1:n` is the set of columns `j`
- `(i, j) ∈ E` whenever `A[i, j] ≠ 0`

A `BipartiteGraph` has two sets of vertices, one for the rows of `A` (which we call side `1`) and one for the columns (which we call side `2`).

# Constructors

    BipartiteGraph(A::SparseMatrixCSC; symmetric_pattern::Bool=false)

When `symmetric_pattern` is `true`, this construction is more efficient.

# Fields

- `S1::SparsityPatternCSC{T}`: maps vertices on side `1` to their neighbors
- `S2::SparsityPatternCSC{T}`: maps vertices on side `2` to their neighbors

# References

> [_What Color Is Your Jacobian? SparsityPatternCSC Coloring for Computing Derivatives_](https://epubs.siam.org/doi/10.1137/S0036144504444711), Gebremedhin et al. (2005)
"""
struct BipartiteGraph{T<:Integer}
    S1::SparsityPatternCSC{T}
    S2::SparsityPatternCSC{T}
end

Base.eltype(::BipartiteGraph{T}) where {T} = T

function BipartiteGraph(A::AbstractMatrix; symmetric_pattern::Bool=false)
    return BipartiteGraph(SparseMatrixCSC(A); symmetric_pattern)
end

function BipartiteGraph(A::SparseMatrixCSC; symmetric_pattern::Bool=false)
    S2 = SparsityPatternCSC(A)  # columns to rows
    if symmetric_pattern
        checksquare(A)  # proxy for checking full symmetry
        S1 = S2
    else
        S1 = transpose(S2)  # rows to columns
    end
    return BipartiteGraph(S1, S2)
end

pattern(bg::BipartiteGraph, ::Val{1}) = bg.S1
pattern(bg::BipartiteGraph, ::Val{2}) = bg.S2

nb_vertices(bg::BipartiteGraph, ::Val{side}) where {side} = pattern(bg, Val(side)).n

nb_edges(bg::BipartiteGraph) = nnz(pattern(bg, Val(1)))

"""
    vertices(bg::BipartiteGraph, Val(side))

Return the list of vertices of `bg` from the specified `side` as a range `1:n`.
"""
vertices(bg::BipartiteGraph, ::Val{side}) where {side} = 1:nb_vertices(bg, Val(side))

"""
    neighbors(bg::BipartiteGraph, Val(side), v::Integer)

Return the neighbors of `v` (a vertex from the specified `side`, `1` or `2`), in the graph `bg`.
"""
function neighbors(bg::BipartiteGraph, ::Val{side}, v::Integer) where {side}
    S = pattern(bg, Val(side))
    return view(rowvals(S), nzrange(S, v))
end

function neighbors_dist2(bg::BipartiteGraph{T}, ::Val{side}, v::Integer) where {T,side}
    # TODO: make more efficient
    other_side = 3 - side
    neigh = Set{T}()
    for u in neighbors(bg, Val(side), v)
        for w in neighbors(bg, Val(other_side), u)
            w != v && push!(neigh, w)
        end
    end
    return neigh
end

function degree(bg::BipartiteGraph, ::Val{side}, v::Integer) where {side}
    return length(neighbors(bg, Val(side), v))
end

function maximum_degree(bg::BipartiteGraph, ::Val{side}) where {side}
    return maximum(v -> degree(bg, Val(side), v), vertices(bg, Val(side)))
end

function minimum_degree(bg::BipartiteGraph, ::Val{side}) where {side}
    return minimum(v -> degree(bg, Val(side), v), vertices(bg, Val(side)))
end

function degree_dist2(bg::BipartiteGraph{T}, ::Val{side}, v::Integer) where {T,side}
    return length(neighbors_dist2(bg, Val(side), v))
end

function has_neighbor_dist2(
    bg::BipartiteGraph, ::Val{side}, v::Integer, u::Integer
) where {side}
    other_side = 3 - side
    for w1 in neighbors(bg, Val(side), v)
        for w2 in neighbors(bg, Val(other_side), w1)
            if w2 == u
                return true
            end
        end
    end
    return false
end

function degree_dist2_in_subset(
    bg::BipartiteGraph, ::Val{side}, v::Integer, subset::AbstractVector{<:Integer}
) where {side}
    d = 0
    for u in subset
        if has_neighbor_dist2(bg, Val(side), v, u)
            d += 1
        end
    end
    return d
end
