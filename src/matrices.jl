const TransposeOrAdjoint{T,M} = Union{Transpose{T,M},Adjoint{T,M}}

"""
    matrix_versions(A::AbstractMatrix)

Return various versions of the same matrix:

- dense and sparse
- transpose and adjoint

Used for internal testing.
"""
function matrix_versions(A)
    A_dense = Matrix(A)
    A_sparse = sparse(A)
    versions = [
        A_dense,
        transpose(Matrix(transpose(A_dense))),
        A_sparse,
        transpose(sparse(transpose(A_sparse))),
    ]
    if issymmetric(A)
        lower_triangles = [
            Matrix(LowerTriangular(A_dense)), sparse(LowerTriangular(A_sparse))
        ]
        upper_triangles = [
            Matrix(UpperTriangular(A_dense)), sparse(UpperTriangular(A_sparse))
        ]
        symmetric_versions = vcat(
            Symmetric.(versions),
            Symmetric.(lower_triangles, :L),
            Symmetric.(upper_triangles, :U),
        )
        append!(versions, symmetric_versions)
    end
    return versions
end

"""
    respectful_similar(A::AbstractMatrix)
    respectful_similar(A::AbstractMatrix, ::Type{T})

Like `Base.similar` but returns a transpose or adjoint when `A` is a transpose or adjoint.
"""
respectful_similar(A::AbstractMatrix) = respectful_similar(A, eltype(A))

respectful_similar(A::AbstractMatrix, ::Type{T}) where {T} = similar(A, T)

function respectful_similar(A::Transpose, ::Type{T}) where {T}
    return transpose(respectful_similar(parent(A), T))
end

function respectful_similar(A::Adjoint, ::Type{T}) where {T}
    return adjoint(respectful_similar(parent(A), T))
end

function respectful_similar(A::Union{Symmetric,Hermitian}, ::Type{T}) where {T}
    return respectful_similar(sparse(A), T)
end

"""
    same_pattern(A, B)

Perform a partial equality check on the sparsity patterns of `A` and `B`:

- if the return is `true`, they might have the same sparsity pattern but we're not sure
- if the return is `false`, they definitely don't have the same sparsity pattern
"""
same_pattern(A, B) = size(A) == size(B)

function same_pattern(
    A::Union{SparseMatrixCSC,SparsityPatternCSC},
    B::Union{SparseMatrixCSC,SparsityPatternCSC},
)
    return size(A) == size(B) && nnz(A) == nnz(B)
end

function check_same_pattern(A, S)
    if !same_pattern(A, S)
        throw(DimensionMismatch("`A` and `S` must have the same sparsity pattern."))
    end
end
