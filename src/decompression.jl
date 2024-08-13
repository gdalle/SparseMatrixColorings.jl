"""
    compress(A, result::AbstractColoringResult)

Compress `A` given a coloring `result` of the sparsity pattern of `A`.

- If `result` comes from a `:column` (resp. `:row`) partition, the output is a single matrix `B` compressed by column (resp. by row).
- If `result` comes from a `:bidirectional` partition, the output is a tuple of matrices `(Br, Bc)`, where `Br` is compressed by row and `Bc` by column.

Compression means summing either the columns or the rows of `A` which share the same color.
It is undone by calling [`decompress`](@ref) or [`decompress!`](@ref).

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
The in-place alternative is [`decompress!`](@ref).

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
The out-of-place alternative is [`decompress`](@ref).

Compression means summing either the columns or the rows of `A` which share the same color.
It is done by calling [`compress`](@ref).

!!! note
    In-place decompression is faster when `A isa SparseMatrixCSC`.

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

## NonSymmetricColoringResult

function decompress_aux!(
    A::AbstractMatrix{R}, B::AbstractMatrix{R}, result::NonSymmetricColoringResult{:column}
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
    A::AbstractMatrix{R}, B::AbstractMatrix{R}, result::NonSymmetricColoringResult{:row}
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
    A::SparseMatrixCSC{R}, B::AbstractMatrix{R}, result::NonSymmetricColoringResult{:column}
) where {R<:Real}
    nzA = nonzeros(A)
    ind = result.compressed_indices
    for i in eachindex(nzA, ind)
        nzA[i] = B[ind[i]]
    end
    return A
end

function decompress_aux!(
    A::SparseMatrixCSC{R}, B::AbstractMatrix{R}, result::NonSymmetricColoringResult{:row}
) where {R<:Real}
    nzA = nonzeros(A)
    ind = result.compressed_indices
    for i in eachindex(nzA, ind)
        nzA[i] = B[ind[i]]
    end
    return A
end

## StarSetColoringResult

function decompress_aux!(
    A::AbstractMatrix{R}, B::AbstractMatrix{R}, result::StarSetColoringResult
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
    A::SparseMatrixCSC{R}, B::AbstractMatrix{R}, result::StarSetColoringResult
) where {R<:Real}
    nzA = nonzeros(A)
    ind = result.compressed_indices
    for i in eachindex(nzA, ind)
        nzA[i] = B[ind[i]]
    end
    return A
end

## TreeSetColoringResult

# TODO: add method for A::SparseMatrixCSC

function decompress_aux!(
    A::AbstractMatrix{R}, B::AbstractMatrix{R}, result::TreeSetColoringResult
) where {R<:Real}
    A .= zero(R)
    S = get_matrix(result)
    color = column_colors(result)
    @compat (; trees, reverse_bfs_orders, stored_values, passed) = result

    if eltype(stored_values) == R
        buffer = stored_values
    else
        buffer = similar(stored_values, R)
    end

    # Recover the diagonal coefficients of A
    for i in axes(A, 1)
        if !iszero(S[i, i])
            A[i, i] = B[i, color[i]]
        end
    end

    # Recover the off-diagonal coefficients of A
    for k in eachindex(trees, reverse_bfs_orders)
        vertices = reverse_bfs_orders[k]
        for vertex in vertices
            buffer[vertex] = zero(R)
            passed[vertex] = false
        end

        tree = trees[k]
        for i in vertices
            for j in tree[i]
                if !passed[j] && !passed[i]
                    val = B[i, color[j]] - buffer[i]
                    buffer[j] = buffer[j] + val
                    A[i, j] = val
                    A[j, i] = val
                    passed[i] = true
                end
            end
        end
    end
    return A
end

## MatrixInverseColoringResult

function decompress_aux!(
    A::AbstractMatrix{R}, B::AbstractMatrix{R}, result::LinearSystemColoringResult
) where {R<:Real}
    S = get_matrix(result)
    color = column_colors(result)
    @compat (; strict_upper_nonzero_inds, T_factorization, strict_upper_nonzeros_A) = result

    # TODO: for some reason I cannot use ldiv! with a sparse QR
    strict_upper_nonzeros_A = T_factorization \ vec(B)
    A .= zero(R)
    for i in axes(A, 1)
        if !iszero(S[i, i])
            A[i, i] = B[i, color[i]]
        end
    end
    for (l, (i, j)) in enumerate(strict_upper_nonzero_inds)
        A[i, j] = strict_upper_nonzeros_A[l]
        A[j, i] = strict_upper_nonzeros_A[l]
    end
    return A
end
