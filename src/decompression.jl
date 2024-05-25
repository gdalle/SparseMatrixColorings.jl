## Column decompression

"""
    decompress_columns!(
        A::AbstractMatrix{R},
        S::AbstractMatrix{Bool},
        C::AbstractMatrix{R},
        colors::AbstractVector{<:Integer}
    ) where {R<:Real}

Decompress the thin matrix `C` into the fat matrix `A` which must have the same sparsity pattern as `S`.

Here, `colors` is a column coloring of `S`, while `C` is a compressed representation of matrix `A` obtained by summing the columns that share the same color.
"""
function decompress_columns! end

function decompress_columns!(
    A::AbstractMatrix{R},
    S::AbstractMatrix{Bool},
    C::AbstractMatrix{R},
    colors::AbstractVector{<:Integer},
) where {R<:Real}
    A .= zero(R)
    for j in axes(A, 2)
        k = colors[j]
        rows_j = (!iszero).(view(S, :, j))
        Aj = view(A, rows_j, j)
        Cj = view(C, rows_j, k)
        copyto!(Aj, Cj)
    end
    return A
end

function decompress_columns!(
    A::SparseMatrixCSC{R},
    S::SparseMatrixCSC{Bool},
    C::AbstractMatrix{R},
    colors::AbstractVector{<:Integer},
) where {R<:Real}
    if !same_sparsity_pattern(A, S)
        throw(DimensionMismatch("`A` and `S` must have the same sparsity pattern."))
    end
    Anz, Arv = nonzeros(A), rowvals(A)
    Anz .= zero(R)
    for j in axes(A, 2)
        k = colors[j]
        nzrange_j = nzrange(A, j)
        rows_j = view(Arv, nzrange_j)
        Aj = view(Anz, nzrange_j)
        Cj = view(C, rows_j, k)
        copyto!(Aj, Cj)
    end
    return A
end

"""
    decompress_columns(
        S::AbstractMatrix{Bool},
        C::AbstractMatrix{R},
        colors::AbstractVector{<:Integer}
    ) where {R<:Real}

Decompress the thin matrix `C` into a new fat matrix `A` with the same sparsity pattern as `S`.

Here, `colors` is a column coloring of `S`, while `C` is a compressed representation of matrix `A` obtained by summing the columns that share the same color.
"""
function decompress_columns(
    S::AbstractMatrix{Bool}, C::AbstractMatrix{R}, colors::AbstractVector{<:Integer}
) where {R<:Real}
    A = respectful_similar(S, R)
    return decompress_columns!(A, S, C, colors)
end

## Row decompression

"""
    decompress_rows!(
        A::AbstractMatrix{R},
        S::AbstractMatrix{Bool},
        C::AbstractMatrix{R},
        colors::AbstractVector{<:Integer}
    ) where {R<:Real}

Decompress the small matrix `C` into the tall matrix `A` which must have the same sparsity pattern as `S`.

Here, `colors` is a row coloring of `S`, while `C` is a compressed representation of matrix `A` obtained by summing the columns that share the same color.
"""
function decompress_rows! end

function decompress_rows!(
    A::AbstractMatrix{R},
    S::AbstractMatrix{Bool},
    C::AbstractMatrix{R},
    colors::AbstractVector{<:Integer},
) where {R<:Real}
    A .= zero(R)
    for i in axes(A, 1)
        k = colors[i]
        cols_i = (!iszero).(view(S, i, :))
        Ai = view(A, i, cols_i)
        Ci = view(C, k, cols_i)
        copyto!(Ai, Ci)
    end
    return A
end

function decompress_rows!(
    A::TransposeOrAdjoint{R,<:SparseMatrixCSC{R}},
    S::TransposeOrAdjoint{Bool,<:SparseMatrixCSC{Bool}},
    C::AbstractMatrix{R},
    colors::AbstractVector{<:Integer},
) where {R<:Real}
    if !same_sparsity_pattern(A, S)
        throw(DimensionMismatch("`A` and `S` must have the same sparsity pattern."))
    end
    PA = parent(A)
    PAnz, PArv = nonzeros(PA), rowvals(PA)
    PAnz .= zero(R)
    for i in axes(A, 1)
        k = colors[i]
        nzrange_i = nzrange(PA, i)
        cols_i = view(PArv, nzrange_i)
        Ai = view(PAnz, nzrange_i)
        Ci = view(C, k, cols_i)
        copyto!(Ai, Ci)
    end
    return A
end

"""
    decompress_rows(
        S::AbstractMatrix{Bool},
        C::AbstractMatrix{R},
        colors::AbstractVector{<:Integer}
    ) where {R<:Real}

Decompress the small matrix `C` into a new tall matrix `A` with the same sparsity pattern as `S`.

Here, `colors` is a row coloring of `S`, while `C` is a compressed representation of matrix `A` obtained by summing the columns that share the same color.
"""
function decompress_rows(
    S::AbstractMatrix{Bool}, C::AbstractMatrix{R}, colors::AbstractVector{<:Integer}
) where {R<:Real}
    A = respectful_similar(S, R)
    return decompress_rows!(A, S, C, colors)
end

## Symmetric decompression

"""
    decompress_symmetric!(
        A::AbstractMatrix{R},
        S::AbstractMatrix{Bool},
        C::AbstractMatrix{R},
        colors::AbstractVector{<:Integer}
    ) where {R<:Real}

Decompress the thin matrix `C` into the symmetric matrix `A` which must have the same sparsity pattern as `S`.

Here, `colors` is a symmetric coloring of `S`, while `C` is a compressed representation of matrix `A` obtained by summing the columns that share the same color.
"""
function decompress_symmetric! end

function decompress_symmetric!(
    A::AbstractMatrix{R},
    S::AbstractMatrix{Bool},
    C::AbstractMatrix{R},
    colors::AbstractVector{<:Integer},
) where {R<:Real}
    A .= zero(R)
    groups = color_groups(colors)
    checksquare(A)
    for i in axes(A, 1), j in axes(A, 2)
        iszero(S[i, j]) && continue
        ki, kj = colors[i], colors[j]
        gi, gj = groups[ki], groups[kj]
        if sum(!iszero, view(S, i, gj)) == 1
            A[i, j] = C[i, kj]
        elseif sum(!iszero, view(S, j, gi)) == 1
            A[i, j] = C[j, ki]
        else
            error("Symmetric coloring is not valid")
        end
    end
    return A
end

function decompress_symmetric!(
    A::Symmetric{R},
    S::AbstractMatrix{Bool},
    C::AbstractMatrix{R},
    colors::AbstractVector{<:Integer},
) where {R<:Real}
    # requires parent decompression to handle both upper and lower triangles
    decompress_symmetric!(parent(A), S, C, colors)
    return A
end

"""
    decompress_symmetric(
        S::AbstractMatrix{Bool},
        C::AbstractMatrix{R},
        colors::AbstractVector{<:Integer}
    ) where {R<:Real}

Decompress the thin matrix `C` into a new symmetric matrix `A` with the same sparsity pattern as `S`.

Here, `colors` is a symmetric coloring of `S`, while `C` is a compressed representation of matrix `A` obtained by summing the columns that share the same color.
"""
function decompress_symmetric(
    S::AbstractMatrix{Bool}, C::AbstractMatrix{R}, colors::AbstractVector{<:Integer}
) where {R<:Real}
    A = respectful_similar(S, R)
    return decompress_symmetric!(A, S, C, colors)
end
