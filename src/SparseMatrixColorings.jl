"""
    SparseMatrixColorings

Coloring algorithms for sparse Jacobian and Hessian matrices.

The algorithms implemented in this package are all taken from the following survey:

> [_What Color Is Your Jacobian? Graph Coloring for Computing Derivatives_](https://epubs.siam.org/doi/10.1137/S0036144504444711), Gebremedhin et al. (2005)

Some parts of the survey (like definitions and theorems) are also copied verbatim or referred to by their number in the documentation.
"""
module SparseMatrixColorings

using ADTypes: ADTypes, AbstractColoringAlgorithm
using LinearAlgebra: Transpose, parent, transpose
using Random: AbstractRNG
using SparseArrays: SparseMatrixCSC, nzrange, rowvals

include("utils.jl")

include("bipartite_graph.jl")
include("adjacency_graph.jl")

include("distance2_coloring.jl")
include("star_coloring.jl")

include("adtypes.jl")
include("check.jl")

export GreedyColoringAlgorithm

end
