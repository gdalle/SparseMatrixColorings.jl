"""
    SparseMatrixColorings

$README
"""
module SparseMatrixColorings

using ADTypes: ADTypes, AbstractColoringAlgorithm
using DocStringExtensions
using LinearAlgebra: Diagonal, Transpose, checksquare, parent, transpose
using Random: AbstractRNG, default_rng, randperm
using SparseArrays:
    SparseArrays,
    SparseMatrixCSC,
    dropzeros,
    dropzeros!,
    nnz,
    nzrange,
    rowvals,
    sparse,
    spzeros

include("graph.jl")
include("order.jl")
include("coloring.jl")
include("adtypes.jl")
include("check.jl")

export GreedyColoringAlgorithm

end
