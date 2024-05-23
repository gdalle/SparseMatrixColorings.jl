"""
    SparseMatrixColorings

$README
"""
module SparseMatrixColorings

using ADTypes: ADTypes, AbstractColoringAlgorithm
using Compat: @compat
using DocStringExtensions: README
using LinearAlgebra: Diagonal, Transpose, checksquare, parent, transpose
using Random: AbstractRNG, default_rng, randperm
using SparseArrays:
    SparseArrays,
    SparseMatrixCSC,
    dropzeros,
    dropzeros!,
    nnz,
    nonzeros,
    nzrange,
    rowvals,
    sparse,
    spzeros

include("graph.jl")
include("order.jl")
include("coloring.jl")
include("adtypes.jl")
include("check.jl")
include("decompression.jl")

@compat public GreedyColoringAlgorithm
@compat public NaturalOrder, RandomOrder, LargestFirst
@compat public decompress_columns!, decompress_columns
@compat public decompress_rows!, decompress_rows
@compat public color_groups

export GreedyColoringAlgorithm

end
