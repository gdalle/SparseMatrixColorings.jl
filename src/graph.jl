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
struct SparsityPatternCSC{Ti<:Integer}
    m::Int
    n::Int
    colptr::Vector{Ti}
    rowval::Vector{Ti}
end

SparsityPatternCSC(A::SparseMatrixCSC) = SparsityPatternCSC(A.m, A.n, A.colptr, A.rowval)

Base.size(S::SparsityPatternCSC) = (S.m, S.n)
Base.size(S::SparsityPatternCSC, d) = d::Integer <= 2 ? size(S)[d] : 1
Base.axes(S::SparsityPatternCSC, d::Integer) = Base.OneTo(size(S, d))

SparseArrays.nnz(S::SparsityPatternCSC) = length(S.rowval)
SparseArrays.rowvals(S::SparsityPatternCSC) = S.rowval
SparseArrays.nzrange(S::SparsityPatternCSC, j::Integer) = S.colptr[j]:(S.colptr[j + 1] - 1)

function SparseArrays.SparseMatrixCSC(S::SparsityPatternCSC)
    return SparseMatrixCSC(S.m, S.n, S.colptr, S.rowval, fill(true, nnz(S)))
end

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

## Adjacency graph

"""
    AbstractAdjacencyGraph{T}

Supertype for various adjacency graph implementations:

- [`AdjacencyGraph`](@ref)
- [`AdjacencyFromBipartiteGraph`](@ref)
"""
abstract type AbstractAdjacencyGraph{T} end

"""
    AdjacencyGraph{T}

Undirected graph without self-loops representing the nonzeros of a symmetric matrix (typically a Hessian matrix).

The adjacency graph of a symmetrix matric `A ∈ ℝ^{n × n}` is `G(A) = (V, E)` where

- `V = 1:n` is the set of rows or columns `i`/`j`
- `(i, j) ∈ E` whenever `A[i, j] ≠ 0` and `i ≠ j`

# Constructors

    AdjacencyGraph(A::SparseMatrixCSC)

# Fields

- `S::SparsityPatternCSC{T}`

# References

> [_What Color Is Your Jacobian? SparsityPatternCSC Coloring for Computing Derivatives_](https://epubs.siam.org/doi/10.1137/S0036144504444711), Gebremedhin et al. (2005)
"""
struct AdjacencyGraph{T} <: AbstractAdjacencyGraph{T}
    S::SparsityPatternCSC{T}
end

AdjacencyGraph(A::AbstractMatrix) = AdjacencyGraph(SparseMatrixCSC(A))
AdjacencyGraph(A::SparseMatrixCSC) = AdjacencyGraph(SparsityPatternCSC(A))

pattern(g::AdjacencyGraph) = g.S
nb_vertices(g::AdjacencyGraph) = pattern(g).n
vertices(g::AbstractAdjacencyGraph) = 1:nb_vertices(g)

function neighbors(g::AdjacencyGraph, v::Integer)
    S = pattern(g)
    neighbors_with_loops = view(rowvals(S), nzrange(S, v))
    return Iterators.filter(!=(v), neighbors_with_loops)  # TODO: optimize
end

function degree(g::AbstractAdjacencyGraph, v::Integer)
    d = 0
    for u in neighbors(g, v)
        if u != v
            d += 1
        end
    end
    return d
end

function nb_edges(g::AbstractAdjacencyGraph)
    ne = 0
    for v in vertices(g)
        for u in neighbors(g, v)
            ne += 1
        end
    end
    return ne ÷ 2
end

maximum_degree(g::AbstractAdjacencyGraph) = maximum(Base.Fix1(degree, g), vertices(g))
minimum_degree(g::AbstractAdjacencyGraph) = minimum(Base.Fix1(degree, g), vertices(g))

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

- `S1::SparsityPatternCSC{T}`: maps vertices on side `1` to their neighbors
- `S2::SparsityPatternCSC{T}`: maps vertices on side `2` to their neighbors

# References

> [_What Color Is Your Jacobian? SparsityPatternCSC Coloring for Computing Derivatives_](https://epubs.siam.org/doi/10.1137/S0036144504444711), Gebremedhin et al. (2005)
"""
struct BipartiteGraph{T<:Integer}
    S1::SparsityPatternCSC{T}
    S2::SparsityPatternCSC{T}
end

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

## Adjacency graph from bipartite

"""
    AdjacencyFromBipartiteGraph{T}

Custom version of [`AdjacencyGraph`](@ref) constructed from a [`BipartiteGraph`](@ref).
If the bipartite graph represents a matrix `A` of size `(m, n)`, then this graph represents the block matrix `[0 A; A' 0]` of size `(n+m) x (n+m)`.

Vertices are ordered as follows:

- from `1` to `n`: column vertices
- from `n+1` to `n+m`: row vertices

# Constructors

    AdjacencyFromBipartiteGraph(A::AbstractMatrix)

# Fields

- `bg::BipartiteGraph{T}`: bipartite graph representation of the matrix `A`
"""
struct AdjacencyFromBipartiteGraph{T} <: AbstractAdjacencyGraph{T}
    bg::BipartiteGraph{T}
end

function AdjacencyFromBipartiteGraph(A::AbstractMatrix; kwargs...)
    return AdjacencyFromBipartiteGraph(BipartiteGraph(A; kwargs...))
end

function nb_vertices(abg::AdjacencyFromBipartiteGraph)
    @compat (; bg) = abg
    m, n = nb_vertices(bg, Val(1)), nb_vertices(bg, Val(2))
    return m + n
end

struct Adder{T}
    y::T
end

(a::Adder)(x) = x + a.y

function neighbors(abg::AdjacencyFromBipartiteGraph, v::Integer)
    @compat (; bg) = abg
    m, n = nb_vertices(bg, Val(1)), nb_vertices(bg, Val(2))
    if 1 <= v <= n
        j = v  # v is a column, it doesn't need shifting
        neigh = neighbors(bg, Val(2), j)
        correction = Adder(n)  # its neighbors are rows, they need shifting
    else
        i = v - n  # v is a row, it needs shifting
        @assert 1 <= i <= m
        neigh = neighbors(bg, Val(1), i)
        correction = Adder(0)  # its neighbors are columns, they don't need shifting
    end
    return Iterators.map(correction, neigh)
end

function pattern(abg::AdjacencyFromBipartiteGraph)
    # TODO: slow
    S = SparseMatrixCSC(pattern(abg.bg, Val(2)))
    m, n = size(S)
    T = eltype(S)
    return SparsityPatternCSC([
        spzeros(T, n, n) transpose(S)
        S spzeros(T, m, m)
    ])
end
