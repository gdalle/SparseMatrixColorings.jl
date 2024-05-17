## Conversion between row- and col-major

"""
    col_major(A::AbstractMatrix)

Construct a column-major representation of the matrix `A`.
"""
col_major(A::M) where {M<:AbstractMatrix} = A
col_major(A::Transpose{<:Any,M}) where {M<:AbstractMatrix} = M(A)

"""
    row_major(A::AbstractMatrix)

Construct a row-major representation of the matrix `A`.
"""
row_major(A::M) where {M<:AbstractMatrix} = transpose(M(transpose(A)))
row_major(A::Transpose{<:Any,M}) where {M<:AbstractMatrix} = A

## Generic nz

function nz_in_col(A_colmajor::AbstractMatrix, j::Integer)
    return filter(i -> !iszero(A_colmajor[i, j]), axes(A_colmajor, 1))
end

function nz_in_row(A_rowmajor::AbstractMatrix, i::Integer)
    return filter(j -> !iszero(A_rowmajor[i, j]), axes(A_rowmajor, 2))
end

## Sparse nz

function nz_in_col(A_colmajor::SparseMatrixCSC{T}, j::Integer) where {T}
    rv = rowvals(A_colmajor)
    ind = nzrange(A_colmajor, j)
    return view(rv, ind)
end

function nz_in_row(A_rowmajor::Transpose{T,<:SparseMatrixCSC{T}}, i::Integer) where {T}
    A_transpose_colmajor = parent(A_rowmajor)
    rv = rowvals(A_transpose_colmajor)
    ind = nzrange(A_transpose_colmajor, i)
    return view(rv, ind)
end
