const TransposeOrAdjoint{T,M} = Union{Transpose{T,M},Adjoint{T,M}}

"""
    matrix_versions(A::AbstractMatrix)

Return various versions of the same matrix:

- dense and sparse
- transpose and adjoint

Used for internal testing.
"""
function matrix_versions(A::AbstractMatrix)
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

"""
    compatible_pattern(A::AbstractMatrix, bg::BipartiteGraph)
    compatible_pattern(A::AbstractMatrix, ag::AdjacencyGraph, uplo::Symbol)

Perform a coarse compatibility check between the sparsity pattern of `A`
and the reference sparsity pattern encoded in a graph structure.

This function only checks necessary conditions for the two sparsity patterns to match.
In particular, it may return `true` even if the patterns are not identical.

When A is a `SparseMatrixCSC`, additional checks on the sparsity structure are performed.

# Return value
- `true`  : the sparsity patterns are potentially compatible
- `false` : the sparsity patterns are definitely incompatible
"""
compatible_pattern(A::AbstractMatrix, bg::BipartiteGraph) = size(A) == size(bg.S2)
function compatible_pattern(A::SparseMatrixCSC, bg::BipartiteGraph)
    size(A) == size(bg.S2) && nnz(A) == nnz(bg.S2)
end

function compatible_pattern(A::AbstractMatrix, ag::AdjacencyGraph, uplo::Symbol)
    size(A) == size(ag.S)
end
function compatible_pattern(A::SparseMatrixCSC, ag::AdjacencyGraph, uplo::Symbol)
    if uplo == :L || uplo == :U
        return size(A) == size(ag.S) && nnz(A) == (nb_edges(ag) + ag.nb_self_loops)
    else
        return size(A) == size(ag.S) && nnz(A) == nnz(ag.S)
    end
end

function check_compatible_pattern(A::AbstractMatrix, bg::BipartiteGraph)
    if !compatible_pattern(A, bg)
        throw(DimensionMismatch("`A` and `bg.S2` must have the same sparsity pattern."))
    end
end

function check_compatible_pattern(A::AbstractMatrix, ag::AdjacencyGraph, uplo::Symbol)
    if !compatible_pattern(A, ag, uplo)
        if uplo == :L
            throw(
                DimensionMismatch(
                    "`A` and `tril(ag.S)` must have the same sparsity pattern."
                ),
            )
        elseif uplo == :U
            throw(
                DimensionMismatch(
                    "`A` and `triu(ag.S)` must have the same sparsity pattern."
                ),
            )
        else  # uplo == :F
            throw(DimensionMismatch("`A` and `ag.S` must have the same sparsity pattern."))
        end
    end
end
