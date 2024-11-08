"""
    compress(A, result::AbstractColoringResult)

Compress `A` given a coloring `result` of the sparsity pattern of `A`.

- If `result` comes from a `:column` (resp. `:row`) partition, the output is a single matrix `B` compressed by column (resp. by row).
- If `result` comes from a `:bidirectional` partition, the output is a tuple of matrices `(Br, Bc)`, where `Br` is compressed by row and `Bc` by column.

Compression means summing either the columns or the rows of `A` which share the same color.
It is undone by calling [`decompress`](@ref) or [`decompress!`](@ref).

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

julia> collect.(column_groups(result))
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

function compress(
    A, result::AbstractColoringResult{structure,:bidirectional}
) where {structure}
    row_group = row_groups(result)
    column_group = column_groups(result)
    Br = stack(row_group; dims=1) do g
        dropdims(sum(A[g, :]; dims=1); dims=1)
    end
    Bc = stack(column_group; dims=2) do g
        dropdims(sum(A[:, g]; dims=2); dims=2)
    end
    return Br, Bc
end

"""
    decompress(B::AbstractMatrix, result::AbstractColoringResult{_,:column/:row})
    decompress(Br::AbstractMatrix, Bc::AbstractMatrix, result::AbstractColoringResult{_,:bidirectional})

Decompress `B` (or the tuple `(Br,Bc)`) into a new matrix `A`, given a coloring `result` of the sparsity pattern of `A`.
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

julia> collect.(column_groups(result))
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
function decompress(B::AbstractMatrix, result::AbstractColoringResult)
    A = respectful_similar(result.A, eltype(B))
    return decompress!(A, B, result)
end

function decompress(
    Br::AbstractMatrix,
    Bc::AbstractMatrix,
    result::AbstractColoringResult{structure,:bidirectional},
) where {structure}
    A = respectful_similar(result.A, Base.promote_eltype(Br, Bc))
    return decompress!(A, Br, Bc, result)
end

"""
    decompress!(
        A::AbstractMatrix, B::AbstractMatrix,
        result::AbstractColoringResult{_,:column/:row}, [uplo=:F]
    )

    decompress!(
        A::AbstractMatrix, Br::AbstractMatrix, Bc::AbstractMatrix
        result::AbstractColoringResult{_,:bidirectional}
    )

Decompress `B` (or the tuple `(Br,Bc)`) in-place into `A`, given a coloring `result` of the sparsity pattern of `A`.
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

julia> collect.(column_groups(result))
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

julia> collect.(column_groups(result))
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
    else  # uplo == :L
        return i >= j
    end
end

## ColumnColoringResult

function decompress!(A::AbstractMatrix, B::AbstractMatrix, result::ColumnColoringResult)
    (; color) = result
    S = result.bg.S2
    check_same_pattern(A, S)
    fill!(A, zero(eltype(A)))
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
    A::AbstractMatrix, b::AbstractVector, c::Integer, result::ColumnColoringResult
)
    (; group) = result
    S = result.bg.S2
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

function decompress!(A::SparseMatrixCSC, B::AbstractMatrix, result::ColumnColoringResult)
    (; compressed_indices) = result
    S = result.bg.S2
    check_same_pattern(A, S)
    nzA = nonzeros(A)
    for k in eachindex(nzA, compressed_indices)
        nzA[k] = B[compressed_indices[k]]
    end
    return A
end

function decompress_single_color!(
    A::SparseMatrixCSC, b::AbstractVector, c::Integer, result::ColumnColoringResult
)
    (; group) = result
    S = result.bg.S2
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

function decompress!(A::AbstractMatrix, B::AbstractMatrix, result::RowColoringResult)
    (; color) = result
    S = result.bg.S2
    check_same_pattern(A, S)
    fill!(A, zero(eltype(A)))
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
    A::AbstractMatrix, b::AbstractVector, c::Integer, result::RowColoringResult
)
    (; group) = result
    S, Sᵀ = result.bg.S2, result.bg.S1
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

function decompress!(A::SparseMatrixCSC, B::AbstractMatrix, result::RowColoringResult)
    (; compressed_indices) = result
    S = result.bg.S2
    check_same_pattern(A, S)
    nzA = nonzeros(A)
    for k in eachindex(nzA, compressed_indices)
        nzA[k] = B[compressed_indices[k]]
    end
    return A
end

## StarSetColoringResult

function decompress!(
    A::AbstractMatrix, B::AbstractMatrix, result::StarSetColoringResult, uplo::Symbol=:F
)
    (; color, star_set) = result
    (; star, hub, spokes) = star_set
    S = result.ag.S
    uplo == :F && check_same_pattern(A, S)
    fill!(A, zero(eltype(A)))
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
    A::AbstractMatrix,
    b::AbstractVector,
    c::Integer,
    result::StarSetColoringResult,
    uplo::Symbol=:F,
)
    (; color, group, star_set) = result
    (; hub, spokes) = star_set
    S = result.ag.S
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
    A::SparseMatrixCSC, B::AbstractMatrix, result::StarSetColoringResult, uplo::Symbol=:F
)
    (; compressed_indices) = result
    S = result.ag.S
    nzA = nonzeros(A)
    if uplo == :F
        check_same_pattern(A, S)
        for k in eachindex(nzA, compressed_indices)
            nzA[k] = B[compressed_indices[k]]
        end
    else
        rvS = rowvals(S)
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
    end
    return A
end

## TreeSetColoringResult

function decompress!(
    A::AbstractMatrix, B::AbstractMatrix, result::TreeSetColoringResult, uplo::Symbol=:F
)
    (; color, vertices_by_tree, reverse_bfs_orders, buffer) = result
    S = result.ag.S
    uplo == :F && check_same_pattern(A, S)
    R = eltype(A)
    fill!(A, zero(R))

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

function decompress!(
    A::SparseMatrixCSC{R},
    B::AbstractMatrix{R},
    result::TreeSetColoringResult,
    uplo::Symbol=:F,
) where {R<:Real}
    (;
        color,
        vertices_by_tree,
        reverse_bfs_orders,
        diagonal_indices,
        diagonal_nzind,
        lower_triangle_offsets,
        upper_triangle_offsets,
        buffer,
    ) = result
    S = result.ag.S
    A_colptr = A.colptr
    nzA = nonzeros(A)
    uplo == :F && check_same_pattern(A, S)

    if eltype(buffer) == R
        buffer_right_type = buffer
    else
        buffer_right_type = similar(buffer, R)
    end

    # Recover the diagonal coefficients of A
    if uplo == :L
        for i in diagonal_indices
            # A[i, i] is the first element in column i
            nzind = A_colptr[i]
            nzA[nzind] = B[i, color[i]]
        end
    elseif uplo == :U
        for i in diagonal_indices
            # A[i, i] is the last element in column i
            nzind = A_colptr[i + 1] - 1
            nzA[nzind] = B[i, color[i]]
        end
    else  # uplo == :F
        for (k, i) in enumerate(diagonal_indices)
            nzind = diagonal_nzind[k]
            nzA[nzind] = B[i, color[i]]
        end
    end

    # Index of offsets in lower_triangle_offsets and upper_triangle_offsets
    counter = 0

    # Recover the off-diagonal coefficients of A
    for k in eachindex(vertices_by_tree, reverse_bfs_orders)
        for vertex in vertices_by_tree[k]
            buffer_right_type[vertex] = zero(R)
        end

        for (i, j) in reverse_bfs_orders[k]
            counter += 1
            val = B[i, color[j]] - buffer_right_type[i]
            buffer_right_type[j] = buffer_right_type[j] + val

            #! format: off
            # A[i,j] is in the lower triangular part of A
            if in_triangle(i, j, :L)
                # uplo = :L or uplo = :F
                # A[i,j] is stored at index_ij = (A.colptr[j+1] - offset_L) in A.nzval
                if uplo != :U
                    nzind = A_colptr[j + 1] - lower_triangle_offsets[counter]
                    nzA[nzind] = val
                end

                # uplo = :U or uplo = :F
                # A[j,i] is stored at index_ji = (A.colptr[i] + offset_U) in A.nzval
                if uplo != :L
                    nzind = A_colptr[i] + upper_triangle_offsets[counter]
                    nzA[nzind] = val
                end

            # A[i,j] is in the upper triangular part of A
            else
                # uplo = :U or uplo = :F
                # A[i,j] is stored at index_ij = (A.colptr[j] + offset_U) in A.nzval
                if uplo != :L
                    nzind = A_colptr[j] + upper_triangle_offsets[counter]
                    nzA[nzind] = val
                end

                # uplo = :L or uplo = :F
                # A[j,i] is stored at index_ji = (A.colptr[i+1] - offset_L) in A.nzval
                if uplo != :U
                    nzind = A_colptr[i + 1] - lower_triangle_offsets[counter]
                    nzA[nzind] = val
                end
            end
            #! format: on
        end
    end
    return A
end

## MatrixInverseColoringResult

function decompress!(
    A::AbstractMatrix,
    B::AbstractMatrix,
    result::LinearSystemColoringResult,
    uplo::Symbol=:F,
)
    (; color, strict_upper_nonzero_inds, T_factorization, strict_upper_nonzeros_A) = result
    S = result.ag.S
    uplo == :F && check_same_pattern(A, S)

    # TODO: for some reason I cannot use ldiv! with a sparse QR
    strict_upper_nonzeros_A = T_factorization \ vec(B)
    fill!(A, zero(eltype(A)))
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

## BicoloringResult

function _join_compressed!(result::BicoloringResult, Br::AbstractMatrix, Bc::AbstractMatrix)
    #=
    Say we have an original matrix `A` of size `(n, m)` and we build an augmented matrix `A_and_Aᵀ = [zeros(n, n) Aᵀ; A zeros(m, m)]`.
    Its first `1:n` columns have the form `[zeros(n); A[:, j]]` and its following `n+1:n+m` columns have the form `[A[i, :]; zeros(m)]`.
    The symmetric column coloring is performed on `A_and_Aᵀ` and the column-wise compression of `A_and_Aᵀ` should return a matrix `Br_and_Bc`.
    But in reality, `Br_and_Bc` is computed as two partial compressions: the row-wise compression `Br` (corresponding to `Aᵀ`) and the columnwise compression `Bc` (corresponding to `A`).
    Before symmetric decompression, we must reconstruct `Br_and_Bc` from `Br` and `Bc`, knowing that the symmetric colors (those making up `Br_and_Bc`) are present in either a row of `Br`, a column of `Bc`, or both.
    Therefore, the column indices in `Br_and_Bc` don't necessarily match with the row indices in `Br` or the column indices in `Bc` since some colors may be missing in the partial compressions.
    The columns of the top part of `Br_and_Bc` (rows `1:n`) are the rows of `Br`, interlaced with zero columns whenever the current color hasn't been used to color any row.
    The columns of the bottom part of `Br_and_Bc` (rows `n+1:n+m`) are the columns of `Bc`, interlaced with zero columns whenever the current color hasn't been used to color any column.
    We use the dictionaries `col_color_ind` and `row_color_ind` to map from symmetric colors to row/column colors.
    =#
    (; A, col_color_ind, row_color_ind) = result
    m, n = size(A)
    R = Base.promote_eltype(Br, Bc)
    if eltype(result.Br_and_Bc) == R
        Br_and_Bc = result.Br_and_Bc
    else
        Br_and_Bc = similar(result.Br_and_Bc, R)
    end
    fill!(Br_and_Bc, zero(R))
    for c in axes(Br_and_Bc, 2)
        if haskey(row_color_ind, c)  # some rows were colored with symmetric color c
            @views copyto!(Br_and_Bc[1:n, c], Br[row_color_ind[c], :])
        end
        if haskey(col_color_ind, c)  # some columns were colored with symmetric c
            @views copyto!(Br_and_Bc[(n + 1):(n + m), c], Bc[:, col_color_ind[c]])
        end
    end
    return Br_and_Bc
end

function decompress!(
    A::AbstractMatrix, Br::AbstractMatrix, Bc::AbstractMatrix, result::BicoloringResult
)
    m, n = size(A)
    Br_and_Bc = _join_compressed!(result, Br, Bc)
    A_and_Aᵀ = decompress(Br_and_Bc, result.symmetric_result)
    copyto!(A, A_and_Aᵀ[(n + 1):(n + m), 1:n])  # original matrix in bottom left corner
    return A
end

function decompress!(
    A::SparseMatrixCSC, Br::AbstractMatrix, Bc::AbstractMatrix, result::BicoloringResult
)
    (; large_colptr, large_rowval, symmetric_result) = result
    m, n = size(A)
    Br_and_Bc = _join_compressed!(result, Br, Bc)
    # pretend A is larger
    A_and_noAᵀ = SparseMatrixCSC(m + n, m + n, large_colptr, large_rowval, A.nzval)
    # decompress lower triangle only
    decompress!(A_and_noAᵀ, Br_and_Bc, symmetric_result, :L)
    return A
end
