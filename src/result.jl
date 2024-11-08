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

Return a vector `group` such that for every color `c`, `group[c]` contains the indices of all columns that are colored with `c`.
"""
function column_groups end

"""
    row_groups(result::AbstractColoringResult)

Return a vector `group` such that for every color `c`, `group[c]` contains the indices of all rows that are colored with `c`.
"""
function row_groups end

"""
    ncolors(result::AbstractColoringResult)

Return the number of different colors used to color the matrix.

For bidirectional partitions, this number is the sum of the number of row colors and the number of column colors.
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
    group_by_color(color::Vector{Int})

Create a color-indexed vector `group` such that `i ∈ group[c]` iff `color[i] == c`.

Assumes the colors are contiguously numbered from `1` to some `cmax`.
"""
function group_by_color(color::AbstractVector{<:Integer})
    cmin, cmax = extrema(color)
    @assert cmin >= 1
    # Compute group sizes and offsets for a joint storage
    group_sizes = zeros(Int, cmax)  # allocation 1, size cmax
    for c in color
        group_sizes[c] += 1
    end
    group_offsets = cumsum(group_sizes)  # allocation 2, size cmax
    # Concatenate all groups inside a single vector
    group_flat = similar(color)  # allocation 3, size n
    for (k, c) in enumerate(color)
        i = group_offsets[c] - group_sizes[c] + 1
        group_flat[i] = k
        group_sizes[c] -= 1
    end
    # Create views into contiguous blocks of the group vector
    group = Vector{typeof(view(group_flat, 1:1))}(undef, cmax)  # allocation 4, size cmax
    for c in 1:cmax
        i = 1 + (c == 1 ? 0 : group_offsets[c - 1])
        j = group_offsets[c]
        group[c] = view(group_flat, i:j)
    end
    return group
end

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
struct ColumnColoringResult{M<:AbstractMatrix,G<:BipartiteGraph,V} <:
       AbstractColoringResult{:nonsymmetric,:column,:direct}
    "matrix that was colored"
    A::M
    "bipartite graph that was used for coloring"
    bg::G
    "one integer color for each column or row (depending on `partition`)"
    color::Vector{Int}
    "color groups for columns or rows (depending on `partition`)"
    group::V
    "flattened indices mapping the compressed matrix `B` to the uncompressed matrix `A` when `A isa SparseMatrixCSC`. They satisfy `nonzeros(A)[k] = vec(B)[compressed_indices[k]]`"
    compressed_indices::Vector{Int}
end

function ColumnColoringResult(A::AbstractMatrix, bg::BipartiteGraph, color::Vector{Int})
    S = bg.S2
    group = group_by_color(color)
    n = size(S, 1)
    rv = rowvals(S)
    compressed_indices = zeros(Int, nnz(S))
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
struct RowColoringResult{M<:AbstractMatrix,G<:BipartiteGraph,V} <:
       AbstractColoringResult{:nonsymmetric,:row,:direct}
    A::M
    bg::G
    color::Vector{Int}
    group::V
    compressed_indices::Vector{Int}
end

function RowColoringResult(A::AbstractMatrix, bg::BipartiteGraph, color::Vector{Int})
    S = bg.S2
    group = group_by_color(color)
    C = length(group)  # ncolors
    rv = rowvals(S)
    compressed_indices = zeros(Int, nnz(S))
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
struct StarSetColoringResult{M<:AbstractMatrix,G<:AdjacencyGraph,V} <:
       AbstractColoringResult{:symmetric,:column,:direct}
    A::M
    ag::G
    color::Vector{Int}
    group::V
    star_set::StarSet
    compressed_indices::Vector{Int}
end

function StarSetColoringResult(
    A::AbstractMatrix, ag::AdjacencyGraph, color::Vector{Int}, star_set::StarSet
)
    S = ag.S
    group = group_by_color(color)
    n = size(S, 1)
    rv = rowvals(S)
    compressed_indices = zeros(Int, nnz(S))
    for j in axes(S, 2)
        for k in nzrange(S, j)
            i = rv[k]
            l, c = symmetric_coefficient(i, j, color, star_set)
            # A[i, j] = B[l, c]
            compressed_indices[k] = (c - 1) * n + l
        end
    end
    return StarSetColoringResult(A, ag, color, group, star_set, compressed_indices)
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
struct TreeSetColoringResult{M<:AbstractMatrix,G<:AdjacencyGraph,V,R} <:
       AbstractColoringResult{:symmetric,:column,:substitution}
    A::M
    ag::G
    color::Vector{Int}
    group::V
    vertices_by_tree::Vector{Vector{Int}}
    reverse_bfs_orders::Vector{Vector{Tuple{Int,Int}}}
    diagonal_indices::Vector{Int}
    diagonal_nzind::Vector{Int}
    lower_triangle_offsets::Vector{Int}
    upper_triangle_offsets::Vector{Int}
    buffer::Vector{R}
end

function TreeSetColoringResult(
    A::AbstractMatrix,
    ag::AdjacencyGraph,
    color::Vector{Int},
    tree_set::TreeSet,
    decompression_eltype::Type{R},
) where {R}
    S = ag.S
    nvertices = length(color)
    group = group_by_color(color)

    # Vector for the decompression of the diagonal coefficients
    diagonal_indices = Int[]
    diagonal_nzind = Int[]
    ndiag = 0

    n = size(S, 1)
    rv = rowvals(S)
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

    # Vectors for the decompression of the off-diagonal coefficients
    nedges = (nnz(S) - ndiag) ÷ 2
    lower_triangle_offsets = Vector{Int}(undef, nedges)
    upper_triangle_offsets = Vector{Int}(undef, nedges)

    # forest is a structure DisjointSets from DataStructures.jl
    # - forest.intmap: a dictionary that maps an edge (i, j) to an integer k
    # - forest.revmap: a dictionary that does the reverse of intmap, mapping an integer k to an edge (i, j)
    # - forest.internal.ngroups: the number of trees in the forest
    forest = tree_set.forest
    ntrees = forest.internal.ngroups

    # dictionary that maps a tree's root to the index of the tree
    roots = Dict{Int,Int}()

    # vector of dictionaries where each dictionary stores the neighbors of each vertex in a tree
    trees = [Dict{Int,Vector{Int}}() for i in 1:ntrees]

    # counter of the number of roots found
    k = 0
    for edge in forest.revmap
        i, j = edge
        # forest has already been compressed so this doesn't change its state
        # I wanted to use find_root but it is deprecated
        root_edge = find_root!(forest, edge)
        root = forest.intmap[root_edge]

        # Update roots
        if !haskey(roots, root)
            k += 1
            roots[root] = k
        end

        # index of the tree T that contains this edge
        index_tree = roots[root]

        # Update the neighbors of i in the tree T
        if !haskey(trees[index_tree], i)
            trees[index_tree][i] = [j]
        else
            push!(trees[index_tree][i], j)
        end

        # Update the neighbors of j in the tree T
        if !haskey(trees[index_tree], j)
            trees[index_tree][j] = [i]
        else
            push!(trees[index_tree][j], i)
        end
    end

    # degrees is a vector of integers that stores the degree of each vertex in a tree
    degrees = Vector{Int}(undef, nvertices)

    # list of vertices for each tree in the forest
    vertices_by_tree = [collect(keys(trees[i])) for i in 1:ntrees]

    # reverse breadth first (BFS) traversal order for each tree in the forest
    reverse_bfs_orders = [Tuple{Int,Int}[] for i in 1:ntrees]

    # nvmax is the number of vertices of the biggest tree in the forest
    nvmax = mapreduce(length, max, vertices_by_tree; init=0)

    # Create a queue with a fixed size nvmax
    queue = Vector{Int}(undef, nvmax)

    # Index in lower_triangle_offsets and upper_triangle_offsets
    index_offsets = 0

    for k in 1:ntrees
        tree = trees[k]

        # Initialize the queue to store the leaves
        queue_start = 1
        queue_end = 0

        # compute the degree of each vertex in the tree
        for (vertex, neighbors) in tree
            degree = length(neighbors)
            degrees[vertex] = degree

            # the vertex is a leaf
            if degree == 1
                queue_end += 1
                queue[queue_end] = vertex
            end
        end

        # continue until all leaves are treated
        while queue_start <= queue_end
            leaf = queue[queue_start]
            queue_start += 1

            # Mark the vertex as removed
            degrees[leaf] = 0

            for neighbor in tree[leaf]
                if degrees[neighbor] != 0
                    # (leaf, neighbor) represents the next edge to visit during decompression
                    push!(reverse_bfs_orders[k], (leaf, neighbor))

                    # reduce the degree of the neighbor
                    degrees[neighbor] -= 1

                    # check if the neighbor is now a leaf
                    if degrees[neighbor] == 1
                        queue_end += 1
                        queue[queue_end] = neighbor
                    end

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
        vertices_by_tree,
        reverse_bfs_orders,
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
struct LinearSystemColoringResult{M<:AbstractMatrix,G<:AdjacencyGraph,V,R,F} <:
       AbstractColoringResult{:symmetric,:column,:substitution}
    A::M
    ag::G
    color::Vector{Int}
    group::V
    strict_upper_nonzero_inds::Vector{Tuple{Int,Int}}
    strict_upper_nonzeros_A::Vector{R}  # TODO: adjust type
    T_factorization::F  # TODO: adjust type
end

function LinearSystemColoringResult(
    A::AbstractMatrix, ag::AdjacencyGraph, color::Vector{Int}, decompression_eltype::Type{R}
) where {R}
    group = group_by_color(color)
    C = length(group)  # ncolors
    S = ag.S
    rv = rowvals(S)

    # build T such that T * strict_upper_nonzeros(A) = B
    # and solve a linear least-squares problem
    # only consider the strict upper triangle of A because of symmetry
    n = checksquare(S)
    strict_upper_nonzero_inds = Tuple{Int,Int}[]
    for j in axes(S, 2)
        for k in nzrange(S, j)
            i = rv[k]
            (i < j) && push!(strict_upper_nonzero_inds, (i, j))
        end
    end

    T = spzeros(float(R), n * C, length(strict_upper_nonzero_inds))
    for (l, (i, j)) in enumerate(strict_upper_nonzero_inds)
        ci = color[i]
        cj = color[j]
        ki = (ci - 1) * n + j  # A[i, j] appears in B[j, ci]
        kj = (cj - 1) * n + i  # A[i, j] appears in B[i, cj]
        T[ki, l] = 1
        T[kj, l] = 1
    end
    T_factorization = factorize(T)

    strict_upper_nonzeros_A = Vector{float(R)}(undef, size(T, 2))

    return LinearSystemColoringResult(
        A,
        ag,
        color,
        group,
        strict_upper_nonzero_inds,
        strict_upper_nonzeros_A,
        T_factorization,
    )
end

## Bicoloring result

"""
    remap_colors(color::Vector{Int})

Renumber the colors in `color` using their index in the vector `sort(unique(color))`, so that they are forced to go from `1` to some `cmax` contiguously.

Return a tuple `(remapped_colors, color_to_ind)` such that `remapped_colors` is a vector containing the renumbered colors and `color_to_ind` is a dictionary giving the translation between old and new color numberings.

For all vertex indices `i` we have:

    remapped_color[i] = color_to_ind[color[i]]
"""
function remap_colors(color::Vector{Int})
    color_to_ind = Dict(c => i for (i, c) in enumerate(sort(unique(color))))
    remapped_colors = [color_to_ind[c] for c in color]
    return remapped_colors, color_to_ind
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
    G<:AdjacencyGraph,
    decompression,
    V,
    SR<:AbstractColoringResult{:symmetric,:column,decompression},
    R,
} <: AbstractColoringResult{:nonsymmetric,:bidirectional,decompression}
    "matrix that was colored"
    A::M
    "adjacency graph that was used for coloring (constructed from the bipartite graph)"
    abg::G
    "one integer color for each column"
    column_color::Vector{Int}
    "one integer color for each row"
    row_color::Vector{Int}
    "color groups for columns"
    column_group::V
    "color groups for rows"
    row_group::V
    "result for the coloring of the symmetric 2x2 block matrix"
    symmetric_result::SR
    "column color to index"
    col_color_ind::Dict{Int,Int}
    "row color to index"
    row_color_ind::Dict{Int,Int}
    "combination of `Br` and `Bc` (almost a concatenation up to color remapping)"
    Br_and_Bc::Matrix{R}
    "CSC storage of `A_and_noAᵀ - `colptr`"
    large_colptr::Vector{Int}
    "CSC storage of `A_and_noAᵀ - `rowval`"
    large_rowval::Vector{Int}
end

column_colors(result::BicoloringResult) = result.column_color
column_groups(result::BicoloringResult) = result.column_group

row_colors(result::BicoloringResult) = result.row_color
row_groups(result::BicoloringResult) = result.row_group

function BicoloringResult(
    A::AbstractMatrix,
    ag::AdjacencyGraph,
    symmetric_result::AbstractColoringResult{:symmetric,:column},
    decompression_eltype::Type{R},
) where {R}
    m, n = size(A)
    symmetric_color = column_colors(symmetric_result)
    column_color, col_color_ind = remap_colors(symmetric_color[1:n])
    row_color, row_color_ind = remap_colors(symmetric_color[(n + 1):(n + m)])
    column_group = group_by_color(column_color)
    row_group = group_by_color(row_color)
    Br_and_Bc = Matrix{R}(undef, n + m, maximum(column_colors(symmetric_result)))
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
        col_color_ind,
        row_color_ind,
        Br_and_Bc,
        large_colptr,
        large_rowval,
    )
end
