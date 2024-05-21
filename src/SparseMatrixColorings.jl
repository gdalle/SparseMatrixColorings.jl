"""
    SparseMatrixColorings

Coloring algorithms for sparse Jacobian and Hessian matrices.

The algorithms implemented in this package are mainly taken from the following papers:

> [_What Color Is Your Jacobian? Graph Coloring for Computing Derivatives_](https://epubs.siam.org/doi/10.1137/S0036144504444711), Gebremedhin et al. (2005)

> [ColPack: Software for graph coloring and related problems in scientific computing](https://dl.acm.org/doi/10.1145/2513109.2513110), Gebremedhin et al. (2013)

Some parts of the articles (like definitions and theorems) are also copied verbatim in the documentation.
"""
module SparseMatrixColorings

using ADTypes: ADTypes, AbstractColoringAlgorithm
using LinearAlgebra: Diagonal, Transpose, checksquare, parent, transpose
using Random: AbstractRNG, default_rng, randperm
using SparseArrays: SparseArrays, SparseMatrixCSC, nzrange, rowvals, spzeros

include("graph.jl")
include("order.jl")
include("coloring.jl")
include("adtypes.jl")
include("check.jl")

export GreedyColoringAlgorithm

end
