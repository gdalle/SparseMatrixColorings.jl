transpose_respecting_similar(A::AbstractMatrix, ::Type{T}) where {T} = similar(A, T)

function transpose_respecting_similar(A::Transpose, ::Type{T}) where {T}
    return transpose(similar(parent(A), T))
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

Decompress the thin matrix `C` into a fat matrix `A` with the same sparsity pattern as `S`.

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
    @views for j in axes(A, 2)
        k = colors[j]
        rows_j = map(!iszero, S[:, j])
        copyto!(A[rows_j, j], C[rows_j, k])
        A[rows_j, j] .= C[rows_j, k]
    end
    return A
end

function decompress_columns!(
    A::SparseMatrixCSC{R},
    S::SparseMatrixCSC{Bool},
    C::AbstractMatrix{R},
    colors::AbstractVector{<:Integer},
) where {R<:Real}
    # assume A and S have the same pattern
    Anz, Arv = nonzeros(A), rowvals(A)
    Anz .= zero(R)
    @views for j in axes(A, 2)
        k = colors[j]
        nzrange_j = nzrange(A, j)
        rows_j = Arv[nzrange_j]
        copyto!(Anz[nzrange_j], C[rows_j, k])
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

Decompress the small matrix `C` into the tall matrix `A` with the same sparsity pattern as `S`.

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
    @views for i in axes(A, 1)
        k = colors[i]
        cols_i = map(!iszero, S[i, :])
        copyto!(A[i, cols_i], C[k, cols_i])
    end
    return A
end

function decompress_rows!(
    A::Transpose{R,<:SparseMatrixCSC{R}},
    S::Transpose{Bool,<:SparseMatrixCSC{Bool}},
    C::AbstractMatrix{R},
    colors::AbstractVector{<:Integer},
) where {R<:Real}
    # assume A and S have the same pattern
    PA = parent(A)
    PAnz, PArv = nonzeros(PA), rowvals(PA)
    PAnz .= zero(R)
    @views for i in axes(A, 1)
        k = colors[i]
        nzrange_i = nzrange(PA, i)
        cols_i = PArv[nzrange_i]
        copyto!(PAnz[nzrange_i], C[k, cols_i])
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
