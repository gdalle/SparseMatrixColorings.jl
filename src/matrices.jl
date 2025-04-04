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
        lower_triangles = [
            Matrix(LowerTriangular(A_dense)), sparse(LowerTriangular(A_sparse))
        ]
        upper_triangles = [
            Matrix(UpperTriangular(A_dense)), sparse(UpperTriangular(A_sparse))
        ]
        symmetric_versions = vcat(
            Symmetric.(versions),
            Hermitian.(versions),
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

function compatible_pattern(
    A::Union{SparseMatrixCSC,SparsityPatternCSC},
    S::Union{SparseMatrixCSC,SparsityPatternCSC};
    allow_denser::Bool=false,
)
    return allow_denser ? nnz(A) >= nnz(S) : nnz(A) == nnz(S)
end

function check_compatible_pattern(A, S; allow_denser::Bool=false)
    if size(A) != size(S)
        msg = "Decompression target must have the same size as the sparsity pattern used for coloring"
        throw(DimensionMismatch(msg))
    elseif !compatible_pattern(A, S; allow_denser)
        msg = """Decompression target must have $(allow_denser ? "at least" : "exactly") as many nonzeros as the sparsity pattern used for coloring"""
        throw(DimensionMismatch(msg))
    end
    return true
end
