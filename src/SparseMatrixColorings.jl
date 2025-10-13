"""
    SparseMatrixColorings

$README

## Exports

$EXPORTS
"""
module SparseMatrixColorings

using ADTypes: ADTypes
using Base.Iterators: Iterators
using DocStringExtensions: README, EXPORTS, SIGNATURES, TYPEDEF, TYPEDFIELDS
using LinearAlgebra:
    Adjoint,
    Diagonal,
    Hermitian,
    LowerTriangular,
    Symmetric,
    Transpose,
    UpperTriangular,
    adjoint,
    checksquare,
    factorize,
    issymmetric,
    ldiv!,
    parent,
    transpose
using PrecompileTools: @compile_workload
using Random: Random, AbstractRNG, default_rng, randperm
using SparseArrays:
    SparseArrays,
    SparseMatrixCSC,
    dropzeros,
    dropzeros!,
    ftranspose!,
    nnz,
    nonzeros,
    nzrange,
    rowvals,
    sparse,
    sprand,
    spzeros

include("graph.jl")
include("forest.jl")
include("order.jl")
include("coloring.jl")
include("result.jl")
include("matrices.jl")
include("interface.jl")
include("constant.jl")
include("adtypes.jl")
include("decompression.jl")
include("check.jl")
include("examples.jl")
include("show_colors.jl")

include("precompile.jl")

export NaturalOrder, RandomOrder, LargestFirst
export DynamicDegreeBasedOrder, SmallestLast, IncidenceDegree, DynamicLargestFirst
export PerfectEliminationOrder
export ColoringProblem, GreedyColoringAlgorithm, AbstractColoringResult
export ConstantColoringAlgorithm
export coloring, fast_coloring
export column_colors, row_colors, ncolors
export column_groups, row_groups
export sparsity_pattern
export compress, decompress, decompress!, decompress_single_color!

end
