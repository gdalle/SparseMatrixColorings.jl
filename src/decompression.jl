"""
    compress(A, result::AbstractColoringResult)

Compress `A` into a new matrix `B`, given a coloring `result` of the sparsity pattern of `A`.

Compression means summing either the columns or the rows of `A` which share the same color.
It is undone by calling [`decompress`](@ref).

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

#=
function decompress_aux!(
    A::AbstractMatrix{R},
    B::AbstractMatrix{R},
    result::AbstractColoringResult{:symmetric,:column,:substitution},
) where {R<:Real}
    @compat (; disjoint_sets, parent) = result

    # to be optimized!
    set_roots = Set{Int}()
    ntrees = 0
    for edge in tree_set.disjoint_sets.revmap
        # ensure that all paths are compressed
        root_edge = find_root!(disjoint_sets, edge)
        root_index = tree_set.disjoint_sets.intmap[root_edge]

        # we exclude trees related to diagonal coefficients
        if (edge[1] != edge[2])
            push!(set_roots, root_index)
        end
    end
    roots = disjoint_sets.internal.parents

    # DEBUG
    println(set_roots)
    ntrees = length(set_roots)
    println(ntrees)

    trees = [Int[] for i in 1:ntrees]
    k = 0
    for root in set_roots
        k += 1
        for (pos, val) in enumerate(roots)
            if root == val
                push!(trees[k], pos)
            end
        end
    end
    # for k in 1:ntrees
    #     nedges = length(trees[k])
    #     if nedges > 1
    #         tree_edges = trees[k]
    #         p = ...
    #         trees[k] = tree_edges[p]
    #     end
    # end

    # DEBUG
    display(trees)

    n = checksquare(A)
    stored_values = Vector{R}(undef, n)
    if !same_sparsity_pattern(A, S)
        throw(DimensionMismatch("`A` and `S` must have the same sparsity pattern."))
    end
    A .= zero(R)
    for i in axes(A, 1)
        if !iszero(S[i, i])
            A[i, i] = B[i, color[i]]
        end
    end
    for tree in trees
        nedges = length(tree)
        if nedges == 1
            edge_index = tree[1]
            i, j = disjoint_sets.revmap[edge_index]
            val = B[i, color[j]]
            A[i, j] = val
            A[j, i] = val
        else
            for edge_index in tree
                i, j = disjoint_sets.revmap[edge_index]
                stored_values[i] = zero(R)
                stored_values[j] = zero(R)
            end
            for edge_index in tree  # edges are sorted by their distance to the root
                i, j = disjoint_sets.revmap[edge_index]
                parent_index = disjoint_sets.internal.parents[edge_index]
                k, l = disjoint_sets.revmap[parent_index]
                # k = parent[edge_index]
                if edge_index != parent_index
                    if i == k || i == l  # vertex i is the parent of vertex j
                        i, j = j, i  # ensure that i always denotes a leaf vertex
                    end
                end
                val = B[i, color[j]] - stored_values[i]
                stored_values[j] = stored_values[j] + val
                A[i, j] = val
                A[j, i] = val
            end
        end
    end
    return A
end
=#
