module SparseMatrixColoringsCliqueTreesExt

import CliqueTrees: permutation, EliminationAlgorithm, MCS
import SparseArrays: SparseMatrixCSC, rowvals, nnz
import SparseMatrixColorings:
    AdjacencyGraph, BipartiteGraph, PerfectEliminationOrder, pattern, vertices

PerfectEliminationOrder() = PerfectEliminationOrder(MCS())

function vertices(g::AdjacencyGraph{T}, order::PerfectEliminationOrder) where {T}
    S = pattern(g)

    # construct matrix with sparsity pattern S
    M = SparseMatrixCSC{Bool,T}(size(S)..., S.colptr, rowvals(S), ones(Bool, nnz(S)))

    # construct a perfect elimination order
    # self-loops are ignored
    order, _ = permutation(M; alg=order.elimination_algorithm)

    return reverse!(order)
end

end # module
