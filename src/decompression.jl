transpose_respecting_similar(A::AbstractMatrix, ::Type{T}) where {T} = similar(A, T)

function transpose_respecting_similar(A::Transpose, ::Type{T}) where {T}
    return transpose(similar(parent(A), T))
end

function same_sparsity_pattern(A::SparseMatrixCSC, B::SparseMatrixCSC)
    if size(A) != size(B)
        return false
    elseif nnz(A) != nnz(B)
        return false
    else
        for j in axes(A, 2)
            rA = nzrange(A, j)
            rB = nzrange(B, j)
            if rA != rB
                return false
            end
            # TODO: check rowvals?
        end
        return true
    end
end

function same_sparsity_pattern(
    A::Transpose{<:Any,<:SparseMatrixCSC}, B::Transpose{<:Any,<:SparseMatrixCSC}
)
    return same_sparsity_pattern(parent(A), parent(B))
end

"""
    color_groups(colors)

Return `groups::Vector{Vector{Int}}` such that `i ∈ groups[c]` iff `colors[i] == c`.

Assumes the colors are contiguously numbered from `1` to some `cmax`.
"""
function color_groups(colors::AbstractVector{<:Integer})
    cmin, cmax = extrema(colors)
    @assert cmin == 1
    groups = [Int[] for c in 1:cmax]
    for (k, c) in enumerate(colors)
        push!(groups[c], k)
    end
    return groups
end

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
    A = transpose_respecting_similar(S, R)
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
    A::Transpose{R,<:SparseMatrixCSC{R}},
    S::Transpose{Bool,<:SparseMatrixCSC{Bool}},
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
    A = transpose_respecting_similar(S, R)
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
    n = checksquare(A)
    for i_and_j in CartesianIndices(A)
        i, j = Tuple(i_and_j)
        i > j && continue
        if iszero(S[i, j])
            continue
        end
        ki, kj = colors[i], colors[j]
        group_i = filter(i2 -> colors[i2] == ki, 1:n)
        group_j = filter(j2 -> colors[j2] == kj, 1:n)
        if sum(!iszero, view(S, i, group_j)) == 1
            A[i, j] = C[j, kj]
        elseif sum(!iszero, view(S, j, group_i)) == 1
            A[i, j] = C[i, ki]
        else
            error("Symmetric coloring is not valid")
        end
    end
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
    A = transpose_respecting_similar(S, R)
    return decompress_symmetric!(A, S, C, colors)
end
