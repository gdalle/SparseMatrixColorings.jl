## Column decompression

"""
    decompress_columns!(
        A::AbstractMatrix{R},
        S::AbstractMatrix{Bool},
        B::AbstractMatrix{R},
        color::AbstractVector{<:Integer}
    ) where {R<:Real}

Decompress the narrow matrix `B` into the wide matrix `A` which must have the same sparsity pattern as `S`.

Here, `color` is a column coloring of `S`, while `B` is a compressed representation of matrix `A` obtained by summing the columns that share the same color.
"""
function decompress_columns! end

function decompress_columns!(
    A::AbstractMatrix{R},
    S::AbstractMatrix{Bool},
    B::AbstractMatrix{R},
    color::AbstractVector{<:Integer},
) where {R<:Real}
    if !same_sparsity_pattern(A, S)
        throw(DimensionMismatch("`A` and `S` must have the same sparsity pattern."))
    end
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
    color::AbstractVector{<:Integer},
) where {R<:Real}
    if !same_sparsity_pattern(A, S)
        throw(DimensionMismatch("`A` and `S` must have the same sparsity pattern."))
    end
    Anz, Arv = nonzeros(A), rowvals(A)
    Anz .= zero(R)
    for j in axes(A, 2)
        cj = color[j]
        nzrange_j = nzrange(A, j)
        rows_j = view(Arv, nzrange_j)
        Aj = view(Anz, nzrange_j)
        Bj = view(B, rows_j, cj)
        copyto!(Aj, Bj)
    end
    return A
end

"""
    decompress_columns(
        S::AbstractMatrix{Bool},
        B::AbstractMatrix{R},
        color::AbstractVector{<:Integer}
    ) where {R<:Real}

Decompress the narrow matrix `B` into a new wide matrix `A` with the same sparsity pattern as `S`.

Here, `color` is a column coloring of `S`, while `B` is a compressed representation of matrix `A` obtained by summing the columns that share the same color.
"""
function decompress_columns(
    S::AbstractMatrix{Bool}, B::AbstractMatrix{R}, color::AbstractVector{<:Integer}
) where {R<:Real}
    A = respectful_similar(S, R)
    return decompress_columns!(A, S, B, color)
end

## Row decompression

"""
    decompress_rows!(
        A::AbstractMatrix{R},
        S::AbstractMatrix{Bool},
        B::AbstractMatrix{R},
        color::AbstractVector{<:Integer}
    ) where {R<:Real}

Decompress the small matrix `B` into the tall matrix `A` which must have the same sparsity pattern as `S`.

Here, `color` is a row coloring of `S`, while `B` is a compressed representation of matrix `A` obtained by summing the columns that share the same color.
"""
function decompress_rows! end

function decompress_rows!(
    A::AbstractMatrix{R},
    S::AbstractMatrix{Bool},
    B::AbstractMatrix{R},
    color::AbstractVector{<:Integer},
) where {R<:Real}
    if !same_sparsity_pattern(A, S)
        throw(DimensionMismatch("`A` and `S` must have the same sparsity pattern."))
    end
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
    A::TransposeOrAdjoint{R,<:SparseMatrixCSC{R}},
    S::TransposeOrAdjoint{Bool,<:SparseMatrixCSC{Bool}},
    B::AbstractMatrix{R},
    color::AbstractVector{<:Integer},
) where {R<:Real}
    if !same_sparsity_pattern(A, S)
        throw(DimensionMismatch("`A` and `S` must have the same sparsity pattern."))
    end
    PA = parent(A)
    PAnz, PArv = nonzeros(PA), rowvals(PA)
    PAnz .= zero(R)
    for i in axes(A, 1)
        ci = color[i]
        nzrange_i = nzrange(PA, i)
        cols_i = view(PArv, nzrange_i)
        Ai = view(PAnz, nzrange_i)
        Bi = view(B, ci, cols_i)
        copyto!(Ai, Bi)
    end
    return A
end

"""
    decompress_rows(
        S::AbstractMatrix{Bool},
        B::AbstractMatrix{R},
        color::AbstractVector{<:Integer}
    ) where {R<:Real}

Decompress the small matrix `B` into a new tall matrix `A` with the same sparsity pattern as `S`.

Here, `color` is a row coloring of `S`, while `B` is a compressed representation of matrix `A` obtained by summing the columns that share the same color.
"""
function decompress_rows(
    S::AbstractMatrix{Bool}, B::AbstractMatrix{R}, color::AbstractVector{<:Integer}
) where {R<:Real}
    A = respectful_similar(S, R)
    return decompress_rows!(A, S, B, color)
end

## Symmetric decompression

"""
    decompress_symmetric!(
        A::AbstractMatrix{R},
        S::AbstractMatrix{Bool},
        B::AbstractMatrix{R},
        color::AbstractVector{<:Integer},
        [star_set::StarSet],
    ) where {R<:Real}

Decompress the narrow matrix `B` into the symmetric matrix `A` which must have the same sparsity pattern as `S`.

Here, `color` is a symmetric coloring of `S`, while `B` is a compressed representation of matrix `A` obtained by summing the columns that share the same color.

Decompression is faster when a [`StarSet`](@ref) is also provided.

# References

> [_Efficient Computation of Sparse Hessians Using Coloring and Automatic Differentiation_](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.1080.0286), Gebremedhin et al. (2009), Figures 2 and 3
"""
function decompress_symmetric! end

function decompress_symmetric!(
    A::AbstractMatrix{R},
    S::AbstractMatrix{Bool},
    B::AbstractMatrix{R},
    color::AbstractVector{<:Integer},
) where {R<:Real}
    checksquare(A)
    if !same_sparsity_pattern(A, S)
        throw(DimensionMismatch("`A` and `S` must have the same sparsity pattern."))
    end
    A .= zero(R)
    group = color_groups(color)
    for ij in findall(!iszero, S)
        i, j = Tuple(ij)
        j2_exists = false
        for j2 in group[color[j]]
            j2 == j && continue
            if !iszero(S[i, j2])
                A[i, j] = A[j, i] = B[j, color[i]]
                j2_exists = true
                break
            end
        end
        if !j2_exists
            A[i, j] = A[j, i] = B[i, color[j]]
        end
    end
    return A
end

function decompress_symmetric!(
    A::AbstractMatrix{R},
    S::AbstractMatrix{Bool},
    B::AbstractMatrix{R},
    color::AbstractVector{<:Integer},
    star_set::StarSet,
) where {R<:Real}
    @compat (; star, hub) = star_set
    checksquare(A)
    if !same_sparsity_pattern(A, S)
        throw(DimensionMismatch("`A` and `S` must have the same sparsity pattern."))
    end
    A .= zero(R)
    for i in axes(A, 1)
        if !iszero(S[i, i])
            A[i, i] = B[i, color[i]]
        end
    end
    for (i, j) in keys(star)
        h = star_set.hub[star[i, j]]
        if h == 0
            # no hub, arbitrary spoke (here i)
            A[i, j] = A[j, i] = B[i, color[i]]
        elseif h == j
            # i is the spoke
            A[i, j] = A[j, i] = B[i, color[h]]
        elseif h == i
            # j is the spoke
            A[i, j] = A[j, i] = B[j, color[h]]
        end
    end
    return A
end

function decompress_symmetric!(
    A::Symmetric{R},
    S::AbstractMatrix{Bool},
    B::AbstractMatrix{R},
    color::AbstractVector{<:Integer},
) where {R<:Real}
    # requires parent decompression to handle both upper and lower triangles
    decompress_symmetric!(parent(A), S, B, color)
    return A
end

function decompress_symmetric!(
    A::Symmetric{R},
    S::AbstractMatrix{Bool},
    B::AbstractMatrix{R},
    color::AbstractVector{<:Integer},
    star_set::StarSet,
) where {R<:Real}
    # requires parent decompression to handle both upper and lower triangles
    decompress_symmetric!(parent(A), S, B, color, star_set)
    return A
end

"""
    decompress_symmetric(
        S::AbstractMatrix{Bool},
        B::AbstractMatrix{R},
        color::AbstractVector{<:Integer},
        [star_set::StarSet],
    ) where {R<:Real}

Decompress the narrow matrix `B` into a new symmetric matrix `A` with the same sparsity pattern as `S`.

Here, `color` is a symmetric coloring of `S`, while `B` is a compressed representation of matrix `A` obtained by summing the columns that share the same color.

Decompression is faster when a [`StarSet`](@ref) is also provided.

# References

> [_Efficient Computation of Sparse Hessians Using Coloring and Automatic Differentiation_](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.1080.0286), Gebremedhin et al. (2009), Figures 2 and 3
"""
function decompress_symmetric(
    S::AbstractMatrix{Bool}, B::AbstractMatrix{R}, color::AbstractVector{<:Integer}
) where {R<:Real}
    A = respectful_similar(S, R)
    return decompress_symmetric!(A, S, B, color)
end

function decompress_symmetric(
    S::AbstractMatrix{Bool},
    B::AbstractMatrix{R},
    color::AbstractVector{<:Integer},
    star_set::StarSet,
) where {R<:Real}
    A = respectful_similar(S, R)
    return decompress_symmetric!(A, S, B, color, star_set)
end
