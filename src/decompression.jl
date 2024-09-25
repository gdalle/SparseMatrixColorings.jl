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
    @compat (; S) = result
    A = respectful_similar(S, R)
    return decompress!(A, B, result)
end

"""
    decompress!(
        A::AbstractMatrix, B::AbstractMatrix,
        result::AbstractColoringResult, [uplo=:F]
    )

Decompress `B` in-place into `A`, given a coloring `result` of the sparsity pattern of `A`.
The out-of-place alternative is [`decompress`](@ref).

!!! note
    In-place decompression is faster when `A isa SparseMatrixCSC`.

Compression means summing either the columns or the rows of `A` which share the same color.
It is done by calling [`compress`](@ref).

For `:symmetric` coloring results (and for those only), an optional positional argument `uplo in (:U, :L, :F)` can be passed to specify which part of the matrix `A` should be updated: the Upper triangle, the Lower triangle, or the Full matrix.
When `A isa SparseMatrixCSC`, using the `uplo` argument requires a target matrix which only stores the relevant triangle(s).

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
function decompress! end

"""
    decompress_single_color!(
        A::AbstractMatrix, b::AbstractVector, c::Integer,
        result::AbstractColoringResult, [uplo=:F]
    )

Decompress the vector `b` corresponding to color `c` in-place into `A`, given a `:direct` coloring `result` of the sparsity pattern of `A` (it will not work with a `:substitution` coloring).

- If `result` comes from a `:nonsymmetric` structure with `:column` partition, this will update the columns of `A` that share color `c` (whose sum makes up `b`).
- If `result` comes from a `:nonsymmetric` structure with `:row` partition, this will update the rows of `A` that share color `c` (whose sum makes up `b`).
- If `result` comes from a `:symmetric` structure with `:column` partition, this will update the coefficients of `A` whose value is deduced from color `c`.

!!! warning
    This function will only update some coefficients of `A`, without resetting the rest to zero.

For `:symmetric` coloring results (and for those only), an optional positional argument `uplo in (:U, :L, :F)` can be passed to specify which part of the matrix `A` should be updated: the Upper triangle, the Lower triangle, or the Full matrix.
When `A isa SparseMatrixCSC`, using the `uplo` argument requires a target matrix which only stores the relevant triangle(s).

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

julia> A2 = similar(A); A2 .= 0;

julia> decompress_single_color!(A2, B[:, 2], 2, result)
4×6 SparseMatrixCSC{Int64, Int64} with 9 stored entries:
 ⋅  ⋅  4  0  ⋅  0
 0  ⋅  ⋅  ⋅  7  ⋅
 ⋅  0  ⋅  ⋅  8  ⋅
 ⋅  0  5  ⋅  ⋅  ⋅

julia> A2[:, [3, 5]] == A[:, [3, 5]]
true
```

# See also

- [`ColoringProblem`](@ref)
- [`AbstractColoringResult`](@ref)
- [`decompress!`](@ref)
"""
function decompress_single_color! end

function in_triangle(i::Integer, j::Integer, uplo::Symbol)
    if uplo == :F
        return true
    elseif uplo == :U
        return i <= j
    else
        return i >= j
    end
end

## ColumnColoringResult

function decompress!(
    A::AbstractMatrix{R}, B::AbstractMatrix{R}, result::ColumnColoringResult
) where {R<:Real}
    @compat (; S, color) = result
    check_same_pattern(A, S)
    A .= zero(R)
    rvS = rowvals(S)
    for j in axes(S, 2)
        cj = color[j]
        for k in nzrange(S, j)
            i = rvS[k]
            A[i, j] = B[i, cj]
        end
    end
    return A
end

function decompress_single_color!(
    A::AbstractMatrix{R}, b::AbstractVector{R}, c::Integer, result::ColumnColoringResult
) where {R<:Real}
    @compat (; S, group) = result
    check_same_pattern(A, S)
    rvS = rowvals(S)
    for j in group[c]
        for k in nzrange(S, j)
            i = rvS[k]
            A[i, j] = b[i]
        end
    end
    return A
end

function decompress!(
    A::SparseMatrixCSC{R}, B::AbstractMatrix{R}, result::ColumnColoringResult
) where {R<:Real}
    @compat (; S, compressed_indices) = result
    check_same_pattern(A, S)
    nzA = nonzeros(A)
    for k in eachindex(nzA, compressed_indices)
        nzA[k] = B[compressed_indices[k]]
    end
    return A
end

function decompress_single_color!(
    A::SparseMatrixCSC{R}, b::AbstractVector{R}, c::Integer, result::ColumnColoringResult
) where {R<:Real}
    @compat (; S, group) = result
    check_same_pattern(A, S)
    rvS = rowvals(S)
    nzA = nonzeros(A)
    for j in group[c]
        for k in nzrange(S, j)
            i = rvS[k]
            nzA[k] = b[i]
        end
    end
    return A
end

## RowColoringResult

function decompress!(
    A::AbstractMatrix{R}, B::AbstractMatrix{R}, result::RowColoringResult
) where {R<:Real}
    @compat (; S, color) = result
    check_same_pattern(A, S)
    A .= zero(R)
    rvS = rowvals(S)
    for j in axes(S, 2)
        for k in nzrange(S, j)
            i = rvS[k]
            ci = color[i]
            A[i, j] = B[ci, j]
        end
    end
    return A
end

function decompress_single_color!(
    A::AbstractMatrix{R}, b::AbstractVector{R}, c::Integer, result::RowColoringResult
) where {R<:Real}
    @compat (; S, Sᵀ, group) = result
    check_same_pattern(A, S)
    rvSᵀ = rowvals(Sᵀ)
    for i in group[c]
        for k in nzrange(Sᵀ, i)
            j = rvSᵀ[k]
            A[i, j] = b[j]
        end
    end
    return A
end

function decompress!(
    A::SparseMatrixCSC{R}, B::AbstractMatrix{R}, result::RowColoringResult
) where {R<:Real}
    @compat (; S, compressed_indices) = result
    check_same_pattern(A, S)
    nzA = nonzeros(A)
    for k in eachindex(nzA, compressed_indices)
        nzA[k] = B[compressed_indices[k]]
    end
    return A
end

function decompress_single_color!(
    A::SparseMatrixCSC{R}, b::AbstractVector{R}, c::Integer, result::RowColoringResult
) where {R<:Real}
    @compat (; S, Sᵀ, group) = result
    check_same_pattern(A, S)
    rvSᵀ = rowvals(Sᵀ)
    nzA = nonzeros(A)
    for i in group[c]
        for k in nzrange(Sᵀ, i)
            j = rvSᵀ[k]
            nzA[k] = b[j]
        end
    end
    return A
end

## StarSetColoringResult

function decompress!(
    A::AbstractMatrix{R},
    B::AbstractMatrix{R},
    result::StarSetColoringResult,
    uplo::Symbol=:F,
) where {R<:Real}
    @compat (; S, color, star_set) = result
    @compat (; star, hub, spokes) = star_set
    uplo == :F && check_same_pattern(A, S)
    A .= zero(R)
    for i in axes(A, 1)
        if !iszero(S[i, i])
            A[i, i] = B[i, color[i]]
        end
    end
    for s in eachindex(hub, spokes)
        j = abs(hub[s])
        cj = color[j]
        for i in spokes[s]
            if in_triangle(i, j, uplo)
                A[i, j] = B[i, cj]
            end
            if in_triangle(j, i, uplo)
                A[j, i] = B[i, cj]
            end
        end
    end
    return A
end

function decompress_single_color!(
    A::AbstractMatrix{R},
    b::AbstractVector{R},
    c::Integer,
    result::StarSetColoringResult,
    uplo::Symbol=:F,
) where {R<:Real}
    @compat (; S, color, group, star_set) = result
    @compat (; hub, spokes) = star_set
    uplo == :F && check_same_pattern(A, S)
    for i in axes(A, 1)
        if !iszero(S[i, i]) && color[i] == c
            A[i, i] = b[i]
        end
    end
    for s in eachindex(hub, spokes)
        j = abs(hub[s])
        if color[j] == c
            for i in spokes[s]
                if in_triangle(i, j, uplo)
                    A[i, j] = b[i]
                end
                if in_triangle(j, i, uplo)
                    A[j, i] = b[i]
                end
            end
        end
    end
    return A
end

function decompress!(
    A::SparseMatrixCSC{R}, B::AbstractMatrix{R}, result::StarSetColoringResult
) where {R<:Real}
    @compat (; S, compressed_indices) = result
    check_same_pattern(A, S)
    nzA = nonzeros(A)
    for k in eachindex(nzA, compressed_indices)
        nzA[k] = B[compressed_indices[k]]
    end
    return A
end

function decompress!(
    A::SparseMatrixCSC{R}, B::AbstractMatrix{R}, result::StarSetColoringResult, uplo::Symbol
) where {R<:Real}
    @compat (; S, compressed_indices) = result
    uplo == :F && check_same_pattern(A, S)
    rvS = rowvals(S)
    nzA = nonzeros(A)
    l = 0  # assume A has the same pattern as the triangle
    for j in axes(S, 2)
        for k in nzrange(S, j)
            i = rvS[k]
            if in_triangle(i, j, uplo)
                l += 1
                nzA[l] = B[compressed_indices[k]]
            end
        end
    end
    @assert l == length(nonzeros(A))
    return A
end

## TreeSetColoringResult

# TODO: add method for A::SparseMatrixCSC

function decompress!(
    A::AbstractMatrix{R},
    B::AbstractMatrix{R},
    result::TreeSetColoringResult,
    uplo::Symbol=:F,
) where {R<:Real}
    @compat (; S, color, vertices_by_tree, reverse_bfs_orders, buffer) = result
    uplo == :F && check_same_pattern(A, S)
    A .= zero(R)

    if eltype(buffer) == R
        buffer_right_type = buffer
    else
        buffer_right_type = similar(buffer, R)
    end

    # Recover the diagonal coefficients of A
    for i in axes(A, 1)
        if !iszero(S[i, i])
            A[i, i] = B[i, color[i]]
        end
    end

    # Recover the off-diagonal coefficients of A
    for k in eachindex(vertices_by_tree, reverse_bfs_orders)
        for vertex in vertices_by_tree[k]
            buffer_right_type[vertex] = zero(R)
        end

        for (i, j) in reverse_bfs_orders[k]
            val = B[i, color[j]] - buffer_right_type[i]
            buffer_right_type[j] = buffer_right_type[j] + val
            if in_triangle(i, j, uplo)
                A[i, j] = val
            end
            if in_triangle(j, i, uplo)
                A[j, i] = val
            end
        end
    end
    return A
end

## MatrixInverseColoringResult

function decompress!(
    A::AbstractMatrix{R},
    B::AbstractMatrix{R},
    result::LinearSystemColoringResult,
    uplo::Symbol=:F,
) where {R<:Real}
    @compat (;
        S, color, strict_upper_nonzero_inds, T_factorization, strict_upper_nonzeros_A
    ) = result
    uplo == :F && check_same_pattern(A, S)

    # TODO: for some reason I cannot use ldiv! with a sparse QR
    strict_upper_nonzeros_A = T_factorization \ vec(B)
    A .= zero(R)
    for i in axes(A, 1)
        if !iszero(S[i, i])
            A[i, i] = B[i, color[i]]
        end
    end
    for (l, (i, j)) in enumerate(strict_upper_nonzero_inds)
        if in_triangle(i, j, uplo)
            A[i, j] = strict_upper_nonzeros_A[l]
        end
        if in_triangle(j, i, uplo)
            A[j, i] = strict_upper_nonzeros_A[l]
        end
    end
    return A
end
