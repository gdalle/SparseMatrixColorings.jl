## Out of place

"""
    decompress_columns(
        S::AbstractMatrix{Bool},
        B::AbstractMatrix{R},
        coloring_result::AbstractColoringResult,
    ) where {R<:Real}

Decompress the columnwise compression `B` into a new matrix `A`, given the sparsity pattern `S` of `A` and a column `coloring_result` of `S`.
"""
function decompress_columns(
    S::AbstractMatrix{Bool}, B::AbstractMatrix{R}, coloring_result::AbstractColoringResult
) where {R<:Real}
    A = respectful_similar(S, R)
    return decompress_columns!(A, S, B, coloring_result)
end

"""
    decompress_rows(
        S::AbstractMatrix{Bool},
        B::AbstractMatrix{R},
        coloring_result::AbstractColoringResult,
    ) where {R<:Real}

Decompress the rowwise compression `B` into a new matrix `A`, given the sparsity pattern `S` of `A` and a row `coloring_result` of `S`.
"""
function decompress_rows(
    S::AbstractMatrix{Bool}, B::AbstractMatrix{R}, coloring_result::AbstractColoringResult
) where {R<:Real}
    A = respectful_similar(S, R)
    return decompress_rows!(A, S, B, coloring_result)
end

"""
    decompress_symmetric(
        S::AbstractMatrix{Bool},
        B::AbstractMatrix{R},
        coloring_result::AbstractColoringResult,
    ) where {R<:Real}

Decompress the columnwise compression `B` into a new matrix `A`, given the sparsity pattern `S` of `A` and a symmetric `coloring_result` of `S`.
"""
function decompress_symmetric(
    S::AbstractMatrix{Bool}, B::AbstractMatrix{R}, coloring_result::AbstractColoringResult
) where {R<:Real}
    A = respectful_similar(S, R)
    return decompress_symmetric!(A, S, B, coloring_result)
end

## Column decompression

"""
    decompress_columns!(
        A::AbstractMatrix{R},
        S::AbstractMatrix{Bool},
        B::AbstractMatrix{R},
        coloring_result::AbstractColoringResult,
    ) where {R<:Real}

Decompress the columnwise compression `B` into `A`, given the sparsity pattern `S` of `A` and a column `coloring_result` of `S`.
"""
function decompress_columns! end

function decompress_columns!(
    A::AbstractMatrix{R},
    S::AbstractMatrix{Bool},
    B::AbstractMatrix{R},
    coloring_result::AbstractColoringResult,
) where {R<:Real}
    if !same_sparsity_pattern(A, S)
        throw(DimensionMismatch("`A` and `S` must have the same sparsity pattern."))
    end
    color = get_colors(coloring_result)
    A .= zero(R)
    for j in axes(A, 2)
        cj = color[j]
        rows_j = (!iszero).(view(S, :, j))
        Aj = view(A, rows_j, j)
        Bj = view(B, rows_j, cj)
        copyto!(Aj, Bj)
    end
    return A
end

function decompress_columns!(
    A::SparseMatrixCSC{R},
    S::SparseMatrixCSC{Bool},
    B::AbstractMatrix{R},
    coloring_result::SparseColoringResult,
) where {R<:Real}
    if !same_sparsity_pattern(A, S)
        throw(DimensionMismatch("`A` and `S` must have the same sparsity pattern."))
    end
    nonzeros(A) .= vec(B)[coloring_result.compressed_indices]
    return A
end

## Row decompression

"""
    decompress_rows!(
        A::AbstractMatrix{R},
        S::AbstractMatrix{Bool},
        B::AbstractMatrix{R},
        coloring_result::AbstractColoringResult,
    ) where {R<:Real}

Decompress the rowwise compression `B` into `A`, given the sparsity pattern `S` of `A` and a row `coloring_result` of `S`.
"""
function decompress_rows! end

function decompress_rows!(
    A::AbstractMatrix{R},
    S::AbstractMatrix{Bool},
    B::AbstractMatrix{R},
    coloring_result::AbstractColoringResult,
) where {R<:Real}
    if !same_sparsity_pattern(A, S)
        throw(DimensionMismatch("`A` and `S` must have the same sparsity pattern."))
    end
    color = get_colors(coloring_result)
    A .= zero(R)
    for i in axes(A, 1)
        ci = color[i]
        cols_i = (!iszero).(view(S, i, :))
        Ai = view(A, i, cols_i)
        Bi = view(B, ci, cols_i)
        copyto!(Ai, Bi)
    end
    return A
end

function decompress_rows!(
    A::SparseMatrixCSC{R},
    S::SparseMatrixCSC{Bool},
    B::AbstractMatrix{R},
    coloring_result::SparseColoringResult,
) where {R<:Real}
    if !same_sparsity_pattern(A, S)
        throw(DimensionMismatch("`A` and `S` must have the same sparsity pattern."))
    end
    nonzeros(A) .= vec(B)[coloring_result.compressed_indices]
    return A
end

## Symmetric decompression

"""
    decompress_symmetric!(
        A::AbstractMatrix{R},
        S::AbstractMatrix{Bool},
        B::AbstractMatrix{R},
        coloring_result::AbstractColoringResult,
    ) where {R<:Real}

Decompress the columnwise compression `B` into `A`, given the sparsity pattern `S` of `A` and a symmetric `coloring_result` of `S`.

# References

> [_Efficient Computation of Sparse Hessians Using Coloring and Automatic Differentiation_](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.1080.0286), Gebremedhin et al. (2009), Figures 2 and 3
"""
function decompress_symmetric! end

function decompress_symmetric!(
    A::AbstractMatrix{R},
    S::AbstractMatrix{Bool},
    B::AbstractMatrix{R},
    coloring_result::AbstractColoringResult,
) where {R<:Real}
    checksquare(A)
    if !same_sparsity_pattern(A, S)
        throw(DimensionMismatch("`A` and `S` must have the same sparsity pattern."))
    end
    color = get_colors(coloring_result)
    group = get_groups(coloring_result)
    A .= zero(R)
    for ij in findall(!iszero, S)
        i, j = Tuple(ij)
        if coloring_result isa SymmetricColoringResult
            k, l = symmetric_coefficient(i, j, color, coloring_result.star_set)
        else
            k, l = symmetric_coefficient(i, j, color, group, S)
        end
        A[i, j] = B[k, l]
    end
    return A
end

function decompress_symmetric!(
    A::SparseMatrixCSC{R},
    S::SparseMatrixCSC{Bool},
    B::AbstractMatrix{R},
    coloring_result::SparseColoringResult,
) where {R<:Real}
    checksquare(A)
    if !same_sparsity_pattern(A, S)
        throw(DimensionMismatch("`A` and `S` must have the same sparsity pattern."))
    end
    B_extract = B[coloring_result.compressed_indices]
    try
        nonzeros(A) .= B_extract
    catch e
        @show size(A) nnz(A) length(nonzeros(A)) length(B_extract)
        throw(e)
    end
    return A
end
