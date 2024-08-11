"""
    compress(A, result::AbstractColoringResult)

Compress `A` given a coloring `result` of the sparsity pattern of `A`.

- If `result` comes from a `:column` (resp. `:row`) partition, the output is a single matrix `B` compressed by column (resp. by row).
- If `result` comes from a `:bidirectional` partition, the output is a tuple of matrices `(Br, Bc)`, where `Br` is compressed by row and `Bc` by column.

Compression means summing either the columns or the rows of `A` which share the same color.
It is undone by calling [`decompress`](@ref).

!!! warning
    At the moment, `:bidirectional` partitions are not implemented.

# Example

```jldoctest
julia> using SparseMatrixColorings, SparseArrays

julia> A = sparse([
           0 0 4 6 0 9
           1 0 0 0 7 0
           0 2 0 0 8 0
           0 3 5 0 0 0
       ]);

julia> result = coloring(A, ColoringProblem(), GreedyColoringAlgorithm());

julia> column_groups(result)
3-element Vector{Vector{Int64}}:
 [1, 2, 4]
 [3, 5]
 [6]

julia> B = compress(A, result)
4×3 Matrix{Int64}:
 6  4  9
 1  7  0
 2  8  0
 3  5  0
```

# See also

- [`ColoringProblem`](@ref)
- [`AbstractColoringResult`](@ref)
"""
function compress end

function compress(A, result::AbstractColoringResult{structure,:column}) where {structure}
    group = column_groups(result)
    B = stack(group; dims=2) do g
        dropdims(sum(A[:, g]; dims=2); dims=2)
    end
    return B
end

function compress(A, result::AbstractColoringResult{structure,:row}) where {structure}
    group = row_groups(result)
    B = stack(group; dims=1) do g
        dropdims(sum(A[g, :]; dims=1); dims=1)
    end
    return B
end

"""
    decompress(B::AbstractMatrix, result::AbstractColoringResult)

Decompress `B` into a new matrix `A`, given a coloring `result` of the sparsity pattern of `A`.

Compression means summing either the columns or the rows of `A` which share the same color.
It is done by calling [`compress`](@ref).

# Example

```jldoctest
julia> using SparseMatrixColorings, SparseArrays

julia> A = sparse([
           0 0 4 6 0 9
           1 0 0 0 7 0
           0 2 0 0 8 0
           0 3 5 0 0 0
       ]);

julia> result = coloring(A, ColoringProblem(), GreedyColoringAlgorithm());

julia> column_groups(result)
3-element Vector{Vector{Int64}}:
 [1, 2, 4]
 [3, 5]
 [6]

julia> B = compress(A, result)
4×3 Matrix{Int64}:
 6  4  9
 1  7  0
 2  8  0
 3  5  0

julia> decompress(B, result)
4×6 SparseMatrixCSC{Int64, Int64} with 9 stored entries:
 ⋅  ⋅  4  6  ⋅  9
 1  ⋅  ⋅  ⋅  7  ⋅
 ⋅  2  ⋅  ⋅  8  ⋅
 ⋅  3  5  ⋅  ⋅  ⋅

julia> decompress(B, result) == A
true
```

# See also

- [`ColoringProblem`](@ref)
- [`AbstractColoringResult`](@ref)
"""
function decompress(B::AbstractMatrix{R}, result::AbstractColoringResult) where {R<:Real}
    S = get_matrix(result)
    A = respectful_similar(S, R)
    return decompress!(A, B, result)
end

"""
    decompress!(
        A::AbstractMatrix, B::AbstractMatrix,
        result::AbstractColoringResult,
    )

Decompress `B` in-place into `A`, given a coloring `result` of the sparsity pattern of `A`.

Compression means summing either the columns or the rows of `A` which share the same color.
It is done by calling [`compress`](@ref).

# Example

```jldoctest
julia> using SparseMatrixColorings, SparseArrays

julia> A = sparse([
           0 0 4 6 0 9
           1 0 0 0 7 0
           0 2 0 0 8 0
           0 3 5 0 0 0
       ]);

julia> result = coloring(A, ColoringProblem(), GreedyColoringAlgorithm());

julia> column_groups(result)
3-element Vector{Vector{Int64}}:
 [1, 2, 4]
 [3, 5]
 [6]

julia> B = compress(A, result)
4×3 Matrix{Int64}:
 6  4  9
 1  7  0
 2  8  0
 3  5  0

julia> A2 = similar(A);

julia> decompress!(A2, B, result)
4×6 SparseMatrixCSC{Int64, Int64} with 9 stored entries:
 ⋅  ⋅  4  6  ⋅  9
 1  ⋅  ⋅  ⋅  7  ⋅
 ⋅  2  ⋅  ⋅  8  ⋅
 ⋅  3  5  ⋅  ⋅  ⋅

julia> A2 == A
true
```

# See also

- [`ColoringProblem`](@ref)
- [`AbstractColoringResult`](@ref)
"""
function decompress!(
    A::AbstractMatrix{R},
    B::AbstractMatrix{R},
    result::AbstractColoringResult{structure,partition,decompression},
) where {R<:Real,structure,partition,decompression}
    # common checks
    S = get_matrix(result)
    structure == :symmetric && checksquare(A)
    if !same_sparsity_pattern(A, S)
        throw(DimensionMismatch("`A` and `S` must have the same sparsity pattern."))
    end
    return decompress_aux!(A, B, result)
end

function decompress_aux!(
    A::AbstractMatrix{R},
    B::AbstractMatrix{R},
    result::AbstractColoringResult{:nonsymmetric,:column,:direct},
) where {R<:Real}
    A .= zero(R)
    S = get_matrix(result)
    color = column_colors(result)
    for j in axes(A, 2)
        cj = color[j]
        rows_j = (!iszero).(view(S, :, j))
        Aj = view(A, rows_j, j)
        Bj = view(B, rows_j, cj)
        copyto!(Aj, Bj)
    end
    return A
end

function decompress_aux!(
    A::AbstractMatrix{R},
    B::AbstractMatrix{R},
    result::AbstractColoringResult{:nonsymmetric,:row,:direct},
) where {R<:Real}
    A .= zero(R)
    S = get_matrix(result)
    color = row_colors(result)
    for i in axes(A, 1)
        ci = color[i]
        cols_i = (!iszero).(view(S, i, :))
        Ai = view(A, i, cols_i)
        Bi = view(B, ci, cols_i)
        copyto!(Ai, Bi)
    end
    return A
end

function decompress_aux!(
    A::AbstractMatrix{R},
    B::AbstractMatrix{R},
    result::AbstractColoringResult{:symmetric,:column,:direct},
) where {R<:Real}
    A .= zero(R)
    S = get_matrix(result)
    color = column_colors(result)
    group = column_groups(result)
    for ij in findall(!iszero, S)
        i, j = Tuple(ij)
        k, l = symmetric_coefficient(i, j, color, group, S)
        A[i, j] = B[k, l]
    end
    return A
end

function decompress_aux!(
    A::AbstractMatrix{R}, B::AbstractMatrix{R}, result::StarSetColoringResult{:column}
) where {R<:Real}
    A .= zero(R)
    S = get_matrix(result)
    color = column_colors(result)
    for ij in findall(!iszero, S)
        i, j = Tuple(ij)
        k, l = symmetric_coefficient(i, j, color, result.star_set)
        A[i, j] = B[k, l]
    end
    return A
end

function decompress_aux!(
    A::AbstractMatrix{R},
    B::AbstractMatrix{R},
    result::AbstractColoringResult{:symmetric,:column,:substitution},
) where {R<:Real}
    # build T such that T * strict_upper_nonzeros(A) = B
    # and solve a linear least-squares problem
    # only consider the strict upper triangle of A because of symmetry
    # TODO: make more efficient
    A .= zero(R)
    S = sparse(get_matrix(result))
    color = column_colors(result)

    n = checksquare(S)
    strict_upper_nonzero_inds = Tuple{Int,Int}[]
    I, J, _ = findnz(S)
    for (i, j) in zip(I, J)
        (i < j) && push!(strict_upper_nonzero_inds, (i, j))
        (i == j) && (A[i, i] = B[i, color[i]])
    end

    T = spzeros(float(R), length(B), length(strict_upper_nonzero_inds))
    for (l, (i, j)) in enumerate(strict_upper_nonzero_inds)
        ci = color[i]
        cj = color[j]
        ki = (ci - 1) * n + j  # A[i, j] appears in B[j, ci]
        kj = (cj - 1) * n + i  # A[i, j] appears in B[i, cj]
        T[ki, l] = one(float(R))
        T[kj, l] = one(float(R))
    end

    strict_upper_nonzeros_A = T \ vec(B)
    for (l, (i, j)) in enumerate(strict_upper_nonzero_inds)
        A[i, j] = strict_upper_nonzeros_A[l]
        A[j, i] = strict_upper_nonzeros_A[l]
    end
    return A
end

function decompress_aux!(
    A::AbstractMatrix{R}, B::AbstractMatrix{R}, result::TreeSetColoringResult
) where {R<:Real}
    n = checksquare(A)
    A .= zero(R)
    S = get_matrix(result)
    color = column_colors(result)

    # forest is a structure DisjointSets from DataStructures.jl
    # - forest.intmap: a dictionary that maps an edge (i, j) to an integer k
    # - forest.revmap: a dictionary that does the reverse of intmap, mapping an integer k to an edge (i, j)
    # - forest.internal.ngroups: the number of trees in the forest
    forest = result.tree_set.forest
    ntrees = forest.internal.ngroups

    # vector of trees where each tree contains the indices of its edges
    trees = [Int[] for i in 1:ntrees]

    # dictionary that maps a tree's root to the index of the tree
    roots = Dict{Int, Int}()

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
                        push!(dfs_orders[k], (i,j))
                    end
                end
            end
        end
    end

    # stored_values holds the sum of edge values for subtrees in a tree.
    # For each vertex i, stored_values[i] is the sum of edge values in the subtree rooted at i.
    stored_values = Vector{R}(undef, n)

    # Recover the diagonal coefficients of A
    for i in axes(A, 1)
        if !iszero(S[i, i])
            A[i, i] = B[i, color[i]]
        end
    end

    # Recover the off-diagonal coefficients of A
    for k in 1:ntrees
        vertices = keys(degrees[k])
        for vertex in vertices
            stored_values[vertex] = zero(R)
        end

        tree = dfs_orders[k]
        for (i, j) in tree
            val = B[i, color[j]] - stored_values[i]
            stored_values[j] = stored_values[j] + val
            A[i, j] = val
            A[j, i] = val
        end
    end
    return A
end
