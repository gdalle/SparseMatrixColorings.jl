module SparseMatrixColoringsCliqueTreesExt

using CliqueTrees: CliqueTrees
using SparseArrays
using SparseMatrixColorings: SparseMatrixColorings, AdjacencyGraph, BipartiteGraph, PerfectEliminationOrder, pattern

function SparseMatrixColorings.vertices(g::AdjacencyGraph{T}, order::PerfectEliminationOrder) where T
    S = pattern(g)

    # construct matrix with sparsity pattern S
    M = SparseMatrixCSC{Bool, T}(size(S)..., S.colptr, rowvals(S), ones(Bool, nnz(S)))

    # can also use alg=CliqueTrees.LexBFS()
    order, _ = CliqueTrees.permutation(M; alg=CliqueTrees.MCS())

    return reverse!(order)
end

function SparseMatrixColorings.vertices(bg::BipartiteGraph{T}, ::Val{side}, order::PerfectEliminationOrder) where {T, side}
    S = pattern(g, Val(side))

    # construct matrix with sparsity pattern S
    M = SparseMatrixCSC{Bool, T}(size(S)..., S.colptr, rowvals(S), ones(Bool, nnz(S)))

    # can also use alg=CliqueTrees.LexBFS()
    order, _ = CliqueTrees.permutation(M; alg=CliqueTrees.MCS())

    return reverse!(order)

end

end # module
