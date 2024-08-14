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
- [`compress`](@ref), [`decompress`](@ref), [`decompress!`](@ref), [`decompress_single_color!`](@ref)

!!! warning
    Unlike the methods above, the concrete subtypes of `AbstractColoringResult` are not part of the public API and may change without notice.
"""
abstract type AbstractColoringResult{structure,partition,decompression,M<:SparseMatrixCSC} end

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
    group_by_color(color::Vector{Int})

Create `group::Vector{Vector{Int}}` such that `i ∈ group[c]` iff `color[i] == c`.

Assumes the colors are contiguously numbered from `1` to some `cmax`.
"""
function group_by_color(color::AbstractVector{<:Integer})
    cmin, cmax = extrema(color)
    @assert cmin == 1
    group = [Int[] for c in 1:cmax]
    for (k, c) in enumerate(color)
        push!(group[c], k)
    end
    return group
end

column_colors(result::AbstractColoringResult{s,:column}) where {s} = result.color
column_groups(result::AbstractColoringResult{s,:column}) where {s} = result.group

row_colors(result::AbstractColoringResult{s,:row}) where {s} = result.color
row_groups(result::AbstractColoringResult{s,:row}) where {s} = result.group

## Concrete subtypes

"""
$TYPEDEF

Storage for the result of a column coloring with direct decompression.

# Fields

$TYPEDFIELDS

# See also

- [`AbstractColoringResult`](@ref)
"""
struct ColumnColoringResult{M} <: AbstractColoringResult{:nonsymmetric,:column,:direct,M}
    "matrix that was colored"
    S::M
    "one integer color for each column or row (depending on `partition`)"
    color::Vector{Int}
    "color groups for columns or rows (depending on `partition`)"
    group::Vector{Vector{Int}}
    "flattened indices mapping the compressed matrix `B` to the uncompressed matrix `A` when `A isa SparseMatrixCSC`. They satisfy `nonzeros(A)[k] = vec(B)[compressed_indices[k]]`"
    compressed_indices::Vector{Int}
end

function ColumnColoringResult(S::SparseMatrixCSC, color::Vector{Int})
    group = group_by_color(color)
    n = size(S, 1)
    I, J, _ = findnz(S)
    compressed_indices = zeros(Int, nnz(S))
    for k in eachindex(I, J, compressed_indices)
        i, j = I[k], J[k]
        c = color[j]
        # A[i, j] = B[i, c]
        compressed_indices[k] = (c - 1) * n + i
    end
    return ColumnColoringResult(S, color, group, compressed_indices)
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
struct RowColoringResult{M} <: AbstractColoringResult{:nonsymmetric,:row,:direct,M}
    S::M
    Sᵀ::M
    color::Vector{Int}
    group::Vector{Vector{Int}}
    compressed_indices::Vector{Int}
end

function RowColoringResult(S::SparseMatrixCSC, color::Vector{Int})
    Sᵀ = sparse(transpose(S))
    group = group_by_color(color)
    C = maximum(color)
    I, J, _ = findnz(S)
    compressed_indices = zeros(Int, nnz(S))
    for k in eachindex(I, J, compressed_indices)
        i, j = I[k], J[k]
        c = color[i]
        # A[i, j] = B[c, j]
        compressed_indices[k] = (j - 1) * C + c
    end
    return RowColoringResult(S, Sᵀ, color, group, compressed_indices)
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
struct StarSetColoringResult{M} <: AbstractColoringResult{:symmetric,:column,:direct,M}
    S::M
    color::Vector{Int}
    group::Vector{Vector{Int}}
    star_set::StarSet
    compressed_indices::Vector{Int}
end

function StarSetColoringResult(S::SparseMatrixCSC, color::Vector{Int}, star_set::StarSet)
    group = group_by_color(color)
    n = size(S, 1)
    I, J, _ = findnz(S)
    compressed_indices = zeros(Int, nnz(S))
    for k in eachindex(I, J, compressed_indices)
        i, j = I[k], J[k]
        l, c = symmetric_coefficient(i, j, color, star_set)
        # A[i, j] = B[l, c]
        compressed_indices[k] = (c - 1) * n + l
    end
    return StarSetColoringResult(S, color, group, star_set, compressed_indices)
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
struct TreeSetColoringResult{M,R} <:
       AbstractColoringResult{:symmetric,:column,:substitution,M}
    S::M
    color::Vector{Int}
    group::Vector{Vector{Int}}
    vertices_by_tree::Vector{Vector{Int}}
    reverse_bfs_orders::Vector{Vector{Tuple{Int,Int}}}
    buffer::Vector{R}
end

function TreeSetColoringResult(
    S::SparseMatrixCSC, color::Vector{Int}, tree_set::TreeSet, decompression_eltype::Type{R}
) where {R}
    nvertices = length(color)
    group = group_by_color(color)

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

    for k in 1:ntrees
        tree = trees[k]

        # queue to store the leaves
        queue = Int[]

        # compute the degree of each vertex in the tree
        for (vertex, neighbors) in trees[k]
            degree = length(neighbors)
            degrees[vertex] = degree

            # the vertex is a leaf
            if degree == 1
                push!(queue, vertex)
            end
        end

        # continue until all leaves are treated
        while !isempty(queue)
            leaf = pop!(queue)

            # Convenient way to specify that the vertex is removed
            degrees[leaf] = 0

            for neighbor in tree[leaf]
                if degrees[neighbor] != 0
                    # (leaf, neighbor) represents the next edge to visit during decompression
                    push!(reverse_bfs_orders[k], (leaf, neighbor))

                    # reduce the degree of all neighbors
                    degrees[neighbor] -= 1

                    # check if the neighbor is now a leaf
                    if degrees[neighbor] == 1
                        push!(queue, neighbor)
                    end
                end
            end
        end
    end

    # buffer holds the sum of edge values for subtrees in a tree.
    # For each vertex i, buffer[i] is the sum of edge values in the subtree rooted at i.
    buffer = Vector{R}(undef, nvertices)

    return TreeSetColoringResult(
        S, color, group, vertices_by_tree, reverse_bfs_orders, buffer
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
struct LinearSystemColoringResult{M,R,F} <:
       AbstractColoringResult{:symmetric,:column,:substitution,M}
    S::M
    color::Vector{Int}
    group::Vector{Vector{Int}}
    strict_upper_nonzero_inds::Vector{Tuple{Int,Int}}
    strict_upper_nonzeros_A::Vector{R}  # TODO: adjust type
    T_factorization::F  # TODO: adjust type
end

function LinearSystemColoringResult(
    S::SparseMatrixCSC, color::Vector{Int}, decompression_eltype::Type{R}
) where {R}
    group = group_by_color(color)
    C = maximum(color)

    # build T such that T * strict_upper_nonzeros(A) = B
    # and solve a linear least-squares problem
    # only consider the strict upper triangle of A because of symmetry
    n = checksquare(S)
    strict_upper_nonzero_inds = Tuple{Int,Int}[]
    I, J, _ = findnz(S)
    for (i, j) in zip(I, J)
        (i < j) && push!(strict_upper_nonzero_inds, (i, j))
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
        S, color, group, strict_upper_nonzero_inds, strict_upper_nonzeros_A, T_factorization
    )
end
