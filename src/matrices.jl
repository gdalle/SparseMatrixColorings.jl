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
        adjoint(Matrix(adjoint(A_dense))),
        A_sparse,
        transpose(sparse(transpose(A_sparse))),
        adjoint(sparse(adjoint(A_sparse))),
    ]
    if issymmetric(A)
        append!(versions, Symmetric.(versions))
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
    return transpose(similar(parent(A), T))
end

function respectful_similar(A::Adjoint, ::Type{T}) where {T}
    return adjoint(similar(parent(A), T))
end

"""
    same_sparsity_pattern(A::AbstractMatrix, B::AbstractMatrix)

Perform a partial equality check on the sparsity patterns of `A` and `B`:

- if the return is `true`, they might have the same sparsity pattern but we're not sure
- if the return is `false`, they definitely don't have the same sparsity pattern
"""
function same_sparsity_pattern(A::AbstractMatrix, B::AbstractMatrix)
    return true
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
    A::TransposeOrAdjoint{<:Any,<:SparseMatrixCSC},
    B::TransposeOrAdjoint{<:Any,<:SparseMatrixCSC},
)
    return same_sparsity_pattern(parent(A), parent(B))
end
