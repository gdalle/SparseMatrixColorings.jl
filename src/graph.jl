## Standard graph

"""
    SparsePatternCSC{Ti<:Integer}

Store a sparse matrix (in CSC) without its values, keeping only the pattern of nonzeros.

# Fields

Copied from `SparseMatrixCSC`:

- `m::Int`: number of rows
- `n::Int`: number of columns
- `colptr::Vector{Ti}`: column `j` is in `colptr[j]:(colptr[j+1]-1)`
- `rowval::Vector{Ti}`: row indices of stored values
"""
struct SparsePatternCSC{Ti<:Integer}
    m::Int
    n::Int
    colptr::Vector{Ti}
    rowval::Vector{Ti}
end

SparsePatternCSC(A::SparseMatrixCSC) = SparsePatternCSC(A.m, A.n, A.colptr, A.rowval)

Base.size(S::SparsePatternCSC) = (S.m, S.n)
SparseArrays.nnz(S::SparsePatternCSC) = length(S.rowval)
SparseArrays.rowvals(S::SparsePatternCSC) = S.rowval
SparseArrays.nzrange(S::SparsePatternCSC, j::Integer) = S.colptr[j]:(S.colptr[j + 1] - 1)

"""
    transpose(S::SparsePatternCSC)

Return a [`SparsePatternCSC`](@ref) corresponding to the transpose of `S`.
"""
function Base.transpose(S::SparsePatternCSC{T}) where {T}
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

    return SparsePatternCSC{T}(n, m, B_colptr, B_rowval)
end

## Adjacency graph

"""
    AdjacencyGraph{T}

Undirected graph without self-loops representing the nonzeros of a symmetric matrix (typically a Hessian matrix).

The adjacency graph of a symmetrix matric `A ∈ ℝ^{n × n}` is `G(A) = (V, E)` where

- `V = 1:n` is the set of rows or columns `i`/`j`
- `(i, j) ∈ E` whenever `A[i, j] ≠ 0` and `i ≠ j`

# Constructors

    AdjacencyGraph(A::SparseMatrixCSC)

# Fields

- `S::SparsePatternCSC{T}`

# References

> [_What Color Is Your Jacobian? SparsePatternCSC Coloring for Computing Derivatives_](https://epubs.siam.org/doi/10.1137/S0036144504444711), Gebremedhin et al. (2005)
"""
struct AdjacencyGraph{T}
    S::SparsePatternCSC{T}
end

AdjacencyGraph(A::SparseMatrixCSC) = AdjacencyGraph(SparsePatternCSC(A))

pattern(g::AdjacencyGraph) = g.S
nb_vertices(g::AdjacencyGraph) = pattern(g).n
vertices(g::AdjacencyGraph) = 1:nb_vertices(g)

function neighbors(g::AdjacencyGraph, v::Integer)
    S = pattern(g)
    neighbors_with_loops = view(rowvals(S), nzrange(S, v))
    return Iterators.filter(!=(v), neighbors_with_loops)  # TODO: optimize
end

function degree(g::AdjacencyGraph, v::Integer)
    d = 0
    for u in neighbors(g, v)
        if u != v
            d += 1
        end
    end
    return d
end

function nb_edges(g::AdjacencyGraph)
    S = pattern(g)
    ne = 0
    for j in vertices(g)
        for k in nzrange(S, j)
            i = rowvals(S)[k]
            if i != j
                ne += 1
            end
        end
    end
    return ne ÷ 2
end

maximum_degree(g::AdjacencyGraph) = maximum(Base.Fix1(degree, g), vertices(g))
minimum_degree(g::AdjacencyGraph) = minimum(Base.Fix1(degree, g), vertices(g))

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

    BipartiteGraph(A::SparseMatrixCSC; symmetric_pattern=false)

When `symmetric_pattern` is `true`, this construction is more efficient.

# Fields

- `S1::SparsePatternCSC{T}`: maps vertices on side `1` to their neighbors
- `S2::SparsePatternCSC{T}`: maps vertices on side `2` to their neighbors

# References

> [_What Color Is Your Jacobian? SparsePatternCSC Coloring for Computing Derivatives_](https://epubs.siam.org/doi/10.1137/S0036144504444711), Gebremedhin et al. (2005)
"""
struct BipartiteGraph{T<:Integer}
    S1::SparsePatternCSC{T}
    S2::SparsePatternCSC{T}
end

function BipartiteGraph(A::SparseMatrixCSC; symmetric_pattern::Bool=false)
    S2 = SparsePatternCSC(A)  # columns to rows
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
    # not efficient, for testing purposes only
    other_side = 3 - side
    neighbors_dist2 = Set{T}()
    for u in neighbors(bg, Val(side), v)
        for w in neighbors(bg, Val(other_side), u)
            w != v && push!(neighbors_dist2, w)
        end
    end
    return length(neighbors_dist2)
end

function maximum_degree_dist2(bg::BipartiteGraph, ::Val{side}) where {side}
    # not efficient, for testing purposes only
    return maximum(v -> degree_dist2(bg, Val(side), v), vertices(bg, Val(side)))
end
