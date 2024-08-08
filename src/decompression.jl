"""
    decompress(B::AbstractMatrix, result::AbstractColoringResult)

Decompress `B` out-of-place into a new matrix `A`, given a coloring `result` of the sparsity pattern of `A`.

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

Decompress `B` in-place into an existing matrix `A`, given a coloring `result` of the sparsity pattern of `A`.

# See also

- [`AbstractColoringResult`](@ref)
"""
function decompress!(
    A::AbstractMatrix{R},
    B::AbstractMatrix{R},
    result::AbstractColoringResult{partition,symmetric,decompression},
) where {R<:Real,partition,symmetric,decompression}
    # common checks
    S = get_matrix(result)
    symmetric && checksquare(A)
    if !same_sparsity_pattern(A, S)
        throw(DimensionMismatch("`A` and `S` must have the same sparsity pattern."))
    end
    return decompress_aux!(A, B, result)
end

## Generic algorithms

function decompress_aux!(
    A::AbstractMatrix{R},
    B::AbstractMatrix{R},
    result::AbstractColoringResult{:column,false,:direct},
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
    result::AbstractColoringResult{:row,false,:direct},
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
    result::AbstractColoringResult{:column,true,:direct},
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

## SparseMatrixCSC

function decompress_aux!(
    A::SparseMatrixCSC{R}, B::AbstractMatrix{R}, result::SparseColoringResult
) where {R<:Real}
    nzA = nonzeros(A)
    ind = result.compressed_indices
    for i in eachindex(nzA, ind)
        nzA[i] = B[ind[i]]
    end
    return A
end
