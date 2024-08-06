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
    findnz,
    nnz,
    nonzeros,
    nzrange,
    rowvals,
    sparse,
    spzeros

include("graph.jl")
include("order.jl")
include("coloring.jl")
include("result.jl")
include("matrices.jl")
include("interface.jl")
include("decompression.jl")
include("check.jl")

@compat public NaturalOrder, RandomOrder, LargestFirst
@compat public AbstractColoringResult
@compat public decompress_columns, decompress_columns!
@compat public decompress_rows, decompress_rows!
@compat public decompress_symmetric, decompress_symmetric!

export GreedyColoringAlgorithm
export column_coloring, row_coloring, symmetric_coloring
export column_coloring_detailed, row_coloring_detailed, symmetric_coloring_detailed
export get_colors, get_groups

end
