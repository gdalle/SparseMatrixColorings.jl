## Abstract type

"""
    AbstractColoringResult{structure,partition,decompression}

Abstract type for the result of a coloring algorithm.

It is the supertype of the object returned by the main function [`coloring`](@ref).

# Type parameters

Combination between the type parameters of [`ColoringProblem`](@ref) and [`GreedyColoringAlgorithm`](@ref):

- `structure::Symbol`: either `:nonsymmetric` or `:symmetric`
- `partition::Symbol`: either `:column`, `:row` or `:bidirectional`
- `decompression::Symbol`: either `:direct` or `:substitution`

# Applicable methods

- [`column_colors`](@ref) and [`column_groups`](@ref) (for a `:column` or `:bidirectional` partition) 
- [`row_colors`](@ref) and [`row_groups`](@ref) (for a `:row` or `:bidirectional` partition)
- [`sparsity_pattern`](@ref)
- [`compress`](@ref), [`decompress`](@ref), [`decompress!`](@ref), [`decompress_single_color!`](@ref)

!!! warning
    Unlike the methods above, the concrete subtypes of `AbstractColoringResult` are not part of the public API and may change without notice.
"""
abstract type AbstractColoringResult{structure,partition,decompression} end

"""
    column_colors(result::AbstractColoringResult)

Return a vector `color` of integer colors, one for each column of the colored matrix.
"""
function column_colors end

"""
    row_colors(result::AbstractColoringResult)

Return a vector `color` of integer colors, one for each row of the colored matrix.
"""
function row_colors end

"""
    column_groups(result::AbstractColoringResult)

Return a vector `group` such that for every non-zero color `c`, `group[c]` contains the indices of all columns that are colored with `c`.
"""
function column_groups end

"""
    row_groups(result::AbstractColoringResult)

Return a vector `group` such that for every non-zero color `c`, `group[c]` contains the indices of all rows that are colored with `c`.
"""
function row_groups end

"""
    ncolors(result::AbstractColoringResult)

Return the number of different non-zero colors used to color the matrix.

For bidirectional partitions, this number is the sum of the number of non-zero row colors and the number of non-zero column colors.
"""
function ncolors(res::AbstractColoringResult{structure,:column}) where {structure}
    return length(column_groups(res))
end

function ncolors(res::AbstractColoringResult{structure,:row}) where {structure}
    return length(row_groups(res))
end

function ncolors(res::AbstractColoringResult{structure,:bidirectional}) where {structure}
    return length(row_groups(res)) + length(column_groups(res))
end

"""
    group_by_color(color::AbstractVector{<:Integer})

Create a color-indexed vector `group` such that `i ∈ group[c]` iff `color[i] == c` for all `c > 0`.

Assumes the colors are contiguously numbered from `0` to some `cmax`.
"""
function group_by_color(::Type{T}, color::AbstractVector) where {T<:Integer}
    cmin, cmax = extrema(color)
    @assert cmin >= 0
    # Compute group sizes and offsets for a joint storage
    group_sizes = zeros(T, cmax)  # allocation 1, size cmax
    for c in color
        if c > 0
            group_sizes[c] += 1
        end
    end
    group_offsets = cumsum(group_sizes)  # allocation 2, size cmax
    # Concatenate all groups inside a single vector
    group_flat = Vector{T}(undef, sum(group_sizes))  # allocation 3, size <= n
    for (k, c) in enumerate(color)
        if c > 0
            i = group_offsets[c] - group_sizes[c] + 1
            group_flat[i] = k
            group_sizes[c] -= 1
        end
    end
    # Create views into contiguous blocks of the group vector
    group = map(1:cmax) do c
        i = 1 + (c == 1 ? 0 : group_offsets[c - 1])
        j = group_offsets[c]
        view(group_flat, i:j)
    end
    return group
end

group_by_color(color::AbstractVector) = group_by_color(Int, color)

const AbstractGroups{T} = AbstractVector{<:AbstractVector{T}}

column_colors(result::AbstractColoringResult{s,:column}) where {s} = result.color
column_groups(result::AbstractColoringResult{s,:column}) where {s} = result.group

row_colors(result::AbstractColoringResult{s,:row}) where {s} = result.color
row_groups(result::AbstractColoringResult{s,:row}) where {s} = result.group

"""
    sparsity_pattern(result::AbstractColoringResult)

Return the matrix that was initially passed to [`coloring`](@ref), without any modifications.

!!! note
    This matrix is not necessarily a `SparseMatrixCSC`, nor does it necessarily have `Bool` entries.
"""
sparsity_pattern(result::AbstractColoringResult) = result.A

## Concrete subtypes

"""
$TYPEDEF

Storage for the result of a column coloring with direct decompression.

# Fields

$TYPEDFIELDS

# See also

- [`AbstractColoringResult`](@ref)
"""
struct ColumnColoringResult{
    M<:AbstractMatrix,T<:Integer,G<:BipartiteGraph{T},GT<:AbstractGroups{T}
} <: AbstractColoringResult{:nonsymmetric,:column,:direct}
    "matrix that was colored"
    A::M
    "bipartite graph that was used for coloring"
    bg::G
    "one integer color for each column or row (depending on `partition`)"
    color::Vector{T}
    "color groups for columns or rows (depending on `partition`)"
    group::GT
    "flattened indices mapping the compressed matrix `B` to the uncompressed matrix `A` when `A isa SparseMatrixCSC`. They satisfy `nonzeros(A)[k] = vec(B)[compressed_indices[k]]`"
    compressed_indices::Vector{T}
end

function ColumnColoringResult(
    A::AbstractMatrix, bg::BipartiteGraph{T}, color::Vector{<:Integer}
) where {T<:Integer}
    S = bg.S2
    group = group_by_color(T, color)
    n = size(S, 1)
    rv = rowvals(S)
    compressed_indices = zeros(T, nnz(S))
    for j in axes(S, 2)
        for k in nzrange(S, j)
            i = rv[k]
            c = color[j]
            # A[i, j] = B[i, c]
            compressed_indices[k] = (c - 1) * n + i
        end
    end
    return ColumnColoringResult(A, bg, color, group, compressed_indices)
end

"""
$TYPEDEF

Storage for the result of a row coloring with direct decompression.

# Fields

See the docstring of [`ColumnColoringResult`](@ref).

$TYPEDFIELDS

# See also

- [`AbstractColoringResult`](@ref)
"""
struct RowColoringResult{
    M<:AbstractMatrix,T<:Integer,G<:BipartiteGraph{T},GT<:AbstractGroups{T}
} <: AbstractColoringResult{:nonsymmetric,:row,:direct}
    A::M
    bg::G
    color::Vector{T}
    group::GT
    compressed_indices::Vector{T}
end

function RowColoringResult(
    A::AbstractMatrix, bg::BipartiteGraph{T}, color::Vector{<:Integer}
) where {T<:Integer}
    S = bg.S2
    group = group_by_color(T, color)
    C = length(group)  # ncolors
    rv = rowvals(S)
    compressed_indices = zeros(T, nnz(S))
    for j in axes(S, 2)
        for k in nzrange(S, j)
            i = rv[k]
            c = color[i]
            # A[i, j] = B[c, j]
            compressed_indices[k] = (j - 1) * C + c
        end
    end
    return RowColoringResult(A, bg, color, group, compressed_indices)
end

"""
$TYPEDEF

Storage for the result of a symmetric coloring with direct decompression.

# Fields

See the docstring of [`ColumnColoringResult`](@ref).

$TYPEDFIELDS

# See also

- [`AbstractColoringResult`](@ref)
"""
struct StarSetColoringResult{
    M<:AbstractMatrix,T<:Integer,G<:AdjacencyGraph{T},GT<:AbstractGroups{T}
} <: AbstractColoringResult{:symmetric,:column,:direct}
    A::M
    ag::G
    color::Vector{T}
    group::GT
    compressed_indices::Vector{T}
end

function StarSetColoringResult(
    A::AbstractMatrix,
    ag::AdjacencyGraph{T},
    color::Vector{<:Integer},
    star_set::StarSet{<:Integer},
) where {T<:Integer}
    (; star, hub) = star_set
    S = pattern(ag)
    edge_to_index = edge_indices(ag)
    n = S.n
    group = group_by_color(T, color)
    rvS = rowvals(S)
    compressed_indices = zeros(T, nnz(S))  # needs to be independent from the storage in the graph, in case the graph gets reused
    for j in axes(S, 2)
        for k in nzrange(S, j)
            i = rvS[k]
            if i == j
                # diagonal coefficients
                c = color[i]
                compressed_indices[k] = (c - 1) * n + i
            else
                # off-diagonal coefficients
                index_ij = edge_to_index[k]
                s = star[index_ij]
                h = abs(hub[s])

                # Assign the non-hub vertex (spoke) to the correct position in spokes
                if i == h
                    # i is the hub and j is the spoke
                    c = color[i]
                    compressed_indices[k] = (c - 1) * n + j
                else  # j == h
                    # j is the hub and i is the spoke
                    c = color[j]
                    compressed_indices[k] = (c - 1) * n + i
                end
            end
        end
    end

    return StarSetColoringResult(A, ag, color, group, compressed_indices)
end

"""
$TYPEDEF

Storage for the result of a symmetric coloring with decompression by substitution.

# Fields

See the docstring of [`ColumnColoringResult`](@ref).

$TYPEDFIELDS

# See also

- [`AbstractColoringResult`](@ref)
"""
struct TreeSetColoringResult{
    M<:AbstractMatrix,T<:Integer,G<:AdjacencyGraph{T},GT<:AbstractGroups{T},R
} <: AbstractColoringResult{:symmetric,:column,:substitution}
    A::M
    ag::G
    color::Vector{T}
    group::GT
    reverse_bfs_orders::Vector{Tuple{T,T}}
    tree_edge_indices::Vector{T}
    nt::T
    diagonal_indices::Vector{T}
    diagonal_nzind::Vector{T}
    lower_triangle_offsets::Vector{T}
    upper_triangle_offsets::Vector{T}
    buffer::Vector{R}
end

function TreeSetColoringResult(
    A::AbstractMatrix,
    ag::AdjacencyGraph{T},
    color::Vector{<:Integer},
    tree_set::TreeSet{<:Integer},
    decompression_eltype::Type{R},
) where {T<:Integer,R}
    (; reverse_bfs_orders, tree_edge_indices, nt) = tree_set
    (; S) = ag
    nvertices = length(color)
    group = group_by_color(T, color)
    rv = rowvals(S)

    # Vector for the decompression of the diagonal coefficients
    diagonal_indices = T[]
    diagonal_nzind = T[]
    ndiag = 0

    if has_diagonal(ag)
        for j in axes(S, 2)
            for k in nzrange(S, j)
                i = rv[k]
                if i == j
                    push!(diagonal_indices, i)
                    push!(diagonal_nzind, k)
                    ndiag += 1
                end
            end
        end
    end

    # Vectors for the decompression of the off-diagonal coefficients
    nedges = (nnz(S) - ndiag) ÷ 2
    lower_triangle_offsets = Vector{T}(undef, nedges)
    upper_triangle_offsets = Vector{T}(undef, nedges)

    # Index in lower_triangle_offsets and upper_triangle_offsets
    index_offsets = 0

    for k in 1:nt
        # Positions of the edges for each tree
        first = tree_edge_indices[k]
        last = tree_edge_indices[k + 1] - 1

        for pos in first:last
            (leaf, neighbor) = reverse_bfs_orders[pos]
            # Update lower_triangle_offsets and upper_triangle_offsets
            i = leaf
            j = neighbor
            col_i = view(rv, nzrange(S, i))
            col_j = view(rv, nzrange(S, j))
            index_offsets += 1

            #! format: off
            # S[i,j] is in the lower triangular part of S
            if in_triangle(i, j, :L)
                # uplo = :L or uplo = :F
                # S[i,j] is stored at index_ij = (S.colptr[j+1] - offset_L) in S.nzval
                lower_triangle_offsets[index_offsets] = length(col_j) - searchsortedfirst(col_j, i) + 1

                # uplo = :U or uplo = :F
                # S[j,i] is stored at index_ji = (S.colptr[i] + offset_U) in S.nzval
                upper_triangle_offsets[index_offsets] = searchsortedfirst(col_i, j)::Int - 1

            # S[i,j] is in the upper triangular part of S
            else
                # uplo = :U or uplo = :F
                # S[i,j] is stored at index_ij = (S.colptr[j] + offset_U) in S.nzval
                upper_triangle_offsets[index_offsets] = searchsortedfirst(col_j, i)::Int - 1

                # uplo = :L or uplo = :F
                # S[j,i] is stored at index_ji = (S.colptr[i+1] - offset_L) in S.nzval
                lower_triangle_offsets[index_offsets] = length(col_i) - searchsortedfirst(col_i, j) + 1
            end
            #! format: on
        end
    end

    # buffer holds the sum of edge values for subtrees in a tree.
    # For each vertex i, buffer[i] is the sum of edge values in the subtree rooted at i.
    buffer = Vector{R}(undef, nvertices)

    return TreeSetColoringResult(
        A,
        ag,
        color,
        group,
        reverse_bfs_orders,
        tree_edge_indices,
        nt,
        diagonal_indices,
        diagonal_nzind,
        lower_triangle_offsets,
        upper_triangle_offsets,
        buffer,
    )
end

## LinearSystemColoringResult

"""
$TYPEDEF

Storage for the result of a symmetric coloring with any decompression.

# Fields

See the docstring of [`ColumnColoringResult`](@ref).

$TYPEDFIELDS

# See also

- [`AbstractColoringResult`](@ref)
"""
struct LinearSystemColoringResult{
    M<:AbstractMatrix,T<:Integer,G<:AdjacencyGraph{T},GT<:AbstractGroups{T},R,F
} <: AbstractColoringResult{:symmetric,:column,:substitution}
    A::M
    ag::G
    color::Vector{T}
    group::GT
    strict_upper_nonzero_inds::Vector{Tuple{T,T}}
    strict_upper_nonzeros_A::Vector{R}  # TODO: adjust type
    M_factorization::F  # TODO: adjust type
end

function LinearSystemColoringResult(
    A::AbstractMatrix,
    ag::AdjacencyGraph{T},
    color::Vector{<:Integer},
    decompression_eltype::Type{R},
) where {T<:Integer,R<:Real}
    group = group_by_color(T, color)
    C = length(group)  # ncolors
    S = ag.S
    rv = rowvals(S)

    # build M such that M * strict_upper_nonzeros(A) = B
    # and solve a linear least-squares problem
    # only consider the strict upper triangle of A because of symmetry
    n = checksquare(S)
    strict_upper_nonzero_inds = Tuple{T,T}[]
    for j in axes(S, 2)
        for k in nzrange(S, j)
            i = rv[k]
            (i < j) && push!(strict_upper_nonzero_inds, (i, j))
        end
    end

    # type annotated because JET was unhappy
    M::SparseMatrixCSC = spzeros(float(R), n * C, length(strict_upper_nonzero_inds))
    for (l, (i, j)) in enumerate(strict_upper_nonzero_inds)
        ci = color[i]
        cj = color[j]
        if ci > 0
            ki = (ci - 1) * n + j  # A[i, j] appears in B[j, ci]
            M[ki, l] = 1
        end
        if cj > 0
            kj = (cj - 1) * n + i  # A[i, j] appears in B[i, cj]
            M[kj, l] = 1
        end
    end
    M_factorization = factorize(M)

    strict_upper_nonzeros_A = Vector{float(R)}(undef, size(M, 2))

    return LinearSystemColoringResult(
        A,
        ag,
        color,
        group,
        strict_upper_nonzero_inds,
        strict_upper_nonzeros_A,
        M_factorization,
    )
end

## Bicoloring result

"""
    remap_colors(color::Vector{<:Integer}, num_sym_colors::Integer, m::Integer, n::Integer)

Return a tuple `(row_color, column_color, symmetric_to_row, symmetric_to_column)` such that `row_color` and `column_color` are vectors containing the renumbered colors for rows and columns.
`symmetric_to_row` and `symmetric_to_column` are vectors that map symmetric colors to row and column colors.

For all vertex indices `i` between `1` and `m` we have:

    row_color[i] = symmetric_to_row[color[n+i]]

For all vertex indices `j` between `1` and `n` we have:

    column_color[j] = symmetric_to_column[color[j]]
"""
function remap_colors(
    ::Type{T}, color::Vector{<:Integer}, num_sym_colors::Integer, m::Integer, n::Integer
) where {T<:Integer}
    # Map symmetric colors to column colors
    symmetric_to_column = zeros(T, num_sym_colors)
    column_color = zeros(T, n)

    counter = 0
    for j in 1:n
        cj = color[j]
        if cj > 0
            # First time that we encounter this column color
            if symmetric_to_column[cj] == 0
                counter += 1
                symmetric_to_column[cj] = counter
            end
            column_color[j] = symmetric_to_column[cj]
        end
    end

    # Map symmetric colors to row colors
    symmetric_to_row = zeros(T, num_sym_colors)
    row_color = zeros(T, m)

    counter = 0
    for i in (n + 1):(n + m)
        ci = color[i]
        if ci > 0
            # First time that we encounter this row color
            if symmetric_to_row[ci] == 0
                counter += 1
                symmetric_to_row[ci] = counter
            end
            row_color[i - n] = symmetric_to_row[ci]
        end
    end

    return row_color, column_color, symmetric_to_row, symmetric_to_column
end

"""
$TYPEDEF

Storage for the result of a bidirectional coloring with direct or substitution decompression, based on the symmetric coloring of a 2x2 block matrix.

# Fields

$TYPEDFIELDS

# See also

- [`AbstractColoringResult`](@ref)
"""
struct BicoloringResult{
    M<:AbstractMatrix,
    T<:Integer,
    G<:AdjacencyGraph{T},
    decompression,
    GT<:AbstractGroups{T},
    SR<:AbstractColoringResult{:symmetric,:column,decompression},
    R,
} <: AbstractColoringResult{:nonsymmetric,:bidirectional,decompression}
    "matrix that was colored"
    A::M
    "augmented adjacency graph that was used for bicoloring"
    abg::G
    "one integer color for each column"
    column_color::Vector{T}
    "one integer color for each row"
    row_color::Vector{T}
    "color groups for columns"
    column_group::GT
    "color groups for rows"
    row_group::GT
    "result for the coloring of the symmetric 2 x 2 block matrix"
    symmetric_result::SR
    "maps symmetric colors to column colors"
    symmetric_to_column::Vector{T}
    "maps symmetric colors to row colors"
    symmetric_to_row::Vector{T}
    "combination of `Br` and `Bc` (almost a concatenation up to color remapping)"
    Br_and_Bc::Matrix{R}
    "CSC storage of `A_and_noAᵀ - `colptr`"
    large_colptr::Vector{T}
    "CSC storage of `A_and_noAᵀ - `rowval`"
    large_rowval::Vector{T}
end

column_colors(result::BicoloringResult) = result.column_color
column_groups(result::BicoloringResult) = result.column_group

row_colors(result::BicoloringResult) = result.row_color
row_groups(result::BicoloringResult) = result.row_group

function BicoloringResult(
    A::AbstractMatrix,
    ag::AdjacencyGraph{T},
    symmetric_result::AbstractColoringResult{:symmetric,:column},
    decompression_eltype::Type{R},
) where {T,R}
    m, n = size(A)
    symmetric_color = column_colors(symmetric_result)
    num_sym_colors = maximum(symmetric_color)
    row_color, column_color, symmetric_to_row, symmetric_to_column = remap_colors(
        T, symmetric_color, num_sym_colors, m, n
    )
    column_group = group_by_color(T, column_color)
    row_group = group_by_color(T, row_color)
    Br_and_Bc = Matrix{R}(undef, n + m, num_sym_colors)
    large_colptr = copy(ag.S.colptr)
    large_colptr[(n + 2):end] .= large_colptr[n + 1]  # last few columns are empty
    large_rowval = ag.S.rowval[1:(end ÷ 2)]  # forget the second half of nonzeros
    return BicoloringResult(
        A,
        ag,
        column_color,
        row_color,
        column_group,
        row_group,
        symmetric_result,
        symmetric_to_column,
        symmetric_to_row,
        Br_and_Bc,
        large_colptr,
        large_rowval,
    )
end
