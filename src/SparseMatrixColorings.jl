"""
    SparseMatrixColorings

$README
"""
module SparseMatrixColorings

using ADTypes:
    ADTypes, AbstractColoringAlgorithm, column_coloring, row_coloring, symmetric_coloring
using Compat: @compat, stack
using DocStringExtensions: README
using LinearAlgebra:
    Adjoint,
    Diagonal,
    Symmetric,
    Transpose,
    adjoint,
    checksquare,
    issymmetric,
    parent,
    transpose
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
include("groups.jl")
include("adtypes.jl")
include("matrices.jl")
include("decompression.jl")
include("check.jl")

@compat public NaturalOrder, RandomOrder, LargestFirst
@compat public color_groups
@compat public decompress_columns, decompress_columns!
@compat public decompress_rows, decompress_rows!
@compat public decompress_symmetric, decompress_symmetric!
@compat public StarSet

export GreedyColoringAlgorithm
export column_coloring, row_coloring, symmetric_coloring, symmetric_coloring_detailed

end
