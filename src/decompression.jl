"""
    decompress(B::AbstractMatrix, coloring_result::AbstractColoringResult)

Decompress `B` out-of-place into a new matrix `A`, given a `coloring_result` of the sparsity pattern of `A`.

# See also

- [`AbstractColoringResult`](@ref)
"""
function decompress(
    B::AbstractMatrix{R}, coloring_result::AbstractColoringResult
) where {R<:Real}
    S = get_matrix(coloring_result)
    A = respectful_similar(S, R)
    return decompress!(A, B, coloring_result)
end

"""
    decompress!(
        A::AbstractMatrix, B::AbstractMatrix,
        coloring_result::AbstractColoringResult,
    )

Decompress `B` in-place into an existing matrix `A`, given a `coloring_result` of the sparsity pattern of `A`.

# See also

- [`AbstractColoringResult`](@ref)
"""
function decompress!(
    A::AbstractMatrix{R},
    B::AbstractMatrix{R},
    coloring_result::AbstractColoringResult{partition,symmetric,decompression},
) where {R<:Real,partition,symmetric,decompression}
    # common checks
    S = get_matrix(coloring_result)
    symmetric && checksquare(A)
    if !same_sparsity_pattern(A, S)
        throw(DimensionMismatch("`A` and `S` must have the same sparsity pattern."))
    end
    A .= zero(R)
    return decompress_aux!(A, B, coloring_result)
end

## Generic algorithms

function decompress_aux!(
    A::AbstractMatrix{R},
    B::AbstractMatrix{R},
    coloring_result::AbstractColoringResult{:column,false,:direct},
) where {R<:Real}
    S = get_matrix(coloring_result)
    color = column_colors(coloring_result)
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
    coloring_result::AbstractColoringResult{:row,false,:direct},
) where {R<:Real}
    S = get_matrix(coloring_result)
    color = row_colors(coloring_result)
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
    coloring_result::AbstractColoringResult{:column,true,:direct},
) where {R<:Real}
    S = get_matrix(coloring_result)
    color = column_colors(coloring_result)
    group = column_groups(coloring_result)
    for ij in findall(!iszero, S)
        i, j = Tuple(ij)
        k, l = symmetric_coefficient(i, j, color, group, S)
        A[i, j] = B[k, l]
    end
    return A
end

## SparseMatrixCSC

function decompress_aux!(
    A::SparseMatrixCSC{R}, B::AbstractMatrix{R}, coloring_result::SparseColoringResult
) where {R<:Real}
    nonzeros(A) .= B[coloring_result.compressed_indices]
    return A
end
