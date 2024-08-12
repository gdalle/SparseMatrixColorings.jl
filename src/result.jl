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
- [`compress`](@ref), [`decompress`](@ref) and [`decompress!`](@ref)

!!! warning
    Unlike the methods above, the concrete subtypes of `AbstractColoringResult` are not part of the public API and may change without notice.
"""
abstract type AbstractColoringResult{structure,partition,decompression,M<:SparseMatrixCSC} end

"""
    get_matrix(result::AbstractColoringResult)

Return the matrix that was colored.
"""
function get_matrix end

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

get_matrix(result::AbstractColoringResult) = result.S

column_colors(result::AbstractColoringResult{s,:column}) where {s} = result.color
column_groups(result::AbstractColoringResult{s,:column}) where {s} = result.group

row_colors(result::AbstractColoringResult{s,:row}) where {s} = result.color
row_groups(result::AbstractColoringResult{s,:row}) where {s} = result.group

## Concrete subtypes

"""
$TYPEDEF

Storage for the result of a nonsymmetric coloring with direct decompression.

# Fields

$TYPEDFIELDS

# See also

- [`AbstractColoringResult`](@ref)
"""
struct NonSymmetricColoringResult{partition,M} <:
       AbstractColoringResult{:nonsymmetric,partition,:direct,M}
    "matrix that was colored"
    S::M
    "one integer color for each column or row (depending on `partition`)"
    color::Vector{Int}
    "color groups for columns or rows (depending on `partition`)"
    group::Vector{Vector{Int}}
    "flattened indices mapping the compressed matrix `B` to the uncompressed matrix `A` when `A isa SparseMatrixCSC`. They satisfy `nonzeros(A)[k] = vec(B)[compressed_indices[k]]`"
    compressed_indices::Vector{Int}
end

function NonSymmetricColoringResult{:column}(S::SparseMatrixCSC, color::Vector{Int})
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
    return NonSymmetricColoringResult{:column,typeof(S)}(
        S, color, group, compressed_indices
    )
end

function NonSymmetricColoringResult{:row}(S::SparseMatrixCSC, color::Vector{Int})
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
    return NonSymmetricColoringResult{:row,typeof(S)}(S, color, group, compressed_indices)
end

"""
$TYPEDEF

Storage for the result of a symmetric coloring with direct decompression.

# Fields

$TYPEDFIELDS

# See also

- [`AbstractColoringResult`](@ref)
- [`NonSymmetricColoringResult`](@ref)
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

$TYPEDFIELDS

# See also

- [`AbstractColoringResult`](@ref)
- [`NonSymmetricColoringResult`](@ref)
"""
struct TreeSetColoringResult{M,R} <:
       AbstractColoringResult{:symmetric,:column,:substitution,M}
    S::M
    color::Vector{Int}
    group::Vector{Vector{Int}}
    tree_set::TreeSet
    degrees::Vector{Dict{Int,Int}}
    dfs_orders::Vector{Vector{Tuple{Int,Int}}}
    stored_values::Vector{R}
end

function TreeSetColoringResult(
    S::SparseMatrixCSC, color::Vector{Int}, tree_set::TreeSet, decompression_eltype::Type{R}
) where {R}
    group = group_by_color(color)

    # forest is a structure DisjointSets from DataStructures.jl
    # - forest.intmap: a dictionary that maps an edge (i, j) to an integer k
    # - forest.revmap: a dictionary that does the reverse of intmap, mapping an integer k to an edge (i, j)
    # - forest.internal.ngroups: the number of trees in the forest
    forest = tree_set.forest
    ntrees = forest.internal.ngroups

    # vector of trees where each tree contains the indices of its edges
    trees = [Int[] for i in 1:ntrees]

    # dictionary that maps a tree's root to the index of the tree
    roots = Dict{Int,Int}()

    k = 0
    for edge in forest.revmap
        root_edge = find_root!(forest, edge)
        root = forest.intmap[root_edge]
        if !haskey(roots, root)
            k += 1
            roots[root] = k
        end
        index_tree = roots[root]
        push!(trees[index_tree], forest.intmap[edge])
    end

    # vector of dictionaries where each dictionary stores the degree of each vertex in a tree
    degrees = [Dict{Int,Int}() for k in 1:ntrees]
    for k in 1:ntrees
        tree = trees[k]
        degree = degrees[k]
        for edge_index in tree
            i, j = forest.revmap[edge_index]
            !haskey(degree, i) && (degree[i] = 0)
            !haskey(degree, j) && (degree[j] = 0)
            degree[i] += 1
            degree[j] += 1
        end
    end

    # depth-first search (DFS) traversal order for each tree in the forest
    dfs_orders = [Vector{Tuple{Int,Int}}() for k in 1:ntrees]
    for k in 1:ntrees
        tree = trees[k]
        degree = degrees[k]
        while sum(values(degree)) != 0
            for (t, edge_index) in enumerate(tree)
                if edge_index != 0
                    i, j = forest.revmap[edge_index]
                    if (degree[i] == 1) || (degree[j] == 1)  # leaf vertex
                        if degree[i] > degree[j]  # vertex i is the parent of vertex j
                            i, j = j, i  # ensure that i always denotes a leaf vertex
                        end
                        degree[i] -= 1  # decrease the degree of vertex i
                        degree[j] -= 1  # decrease the degree of vertex j
                        tree[t] = 0  # remove the edge (i,j)
                        push!(dfs_orders[k], (i, j))
                    end
                end
            end
        end
    end

    n = checksquare(S)
    stored_values = Vector{R}(undef, n)

    return TreeSetColoringResult(
        S, color, group, tree_set, degrees, dfs_orders, stored_values
    )
end

## LinearSystemColoringResult

"""
$TYPEDEF

Storage for the result of a symmetric coloring with any decompression.

# Fields

$TYPEDFIELDS

# See also

- [`AbstractColoringResult`](@ref)
- [`NonSymmetricColoringResult`](@ref)
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

    T = spzeros(Float64, n * C, length(strict_upper_nonzero_inds))
    for (l, (i, j)) in enumerate(strict_upper_nonzero_inds)
        ci = color[i]
        cj = color[j]
        ki = (ci - 1) * n + j  # A[i, j] appears in B[j, ci]
        kj = (cj - 1) * n + i  # A[i, j] appears in B[i, cj]
        T[ki, l] = 1
        T[kj, l] = 1
    end
    T_factorization = factorize(T)

    strict_upper_nonzeros_A = Vector{R}(undef, size(T, 2))

    return LinearSystemColoringResult(
        S, color, group, strict_upper_nonzero_inds, strict_upper_nonzeros_A, T_factorization
    )
end
