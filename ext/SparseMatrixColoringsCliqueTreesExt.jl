module SparseMatrixColoringsCliqueTreesExt

import CliqueTrees: permutation, LexBFS, MCS
import SparseArrays: SparseMatrixCSC, rowvals, nnz
import SparseMatrixColorings:
    AdjacencyGraph, BipartiteGraph, PerfectEliminationOrder, pattern, vertices

function vertices(g::AdjacencyGraph{T}, order::PerfectEliminationOrder) where {T}
    S = pattern(g)

    # construct matrix with sparsity pattern S
    M = SparseMatrixCSC{Bool,T}(size(S)..., S.colptr, rowvals(S), ones(Bool, nnz(S)))

    # construct a perfect elimination order
    # self-loops are ignored
    # we can also use alg=LexBFS()
    # - time complexity: O(|V| + |E|)
    # - space complexity: O(|V| + |E|)
    order, _ = permutation(M; alg=MCS())

    return reverse!(order)
end

end # module
