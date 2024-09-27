"""
    SparseMatrixColorings

$README

## Exports

$EXPORTS
"""
module SparseMatrixColorings

using ADTypes: ADTypes
using Base.Iterators: Iterators
using Compat: @compat, stack
using DataStructures: DisjointSets, find_root!, root_union!, num_groups
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
using Random: AbstractRNG, default_rng, randperm
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
    spzeros

include("graph.jl")
include("order.jl")
include("coloring.jl")
include("result.jl")
include("matrices.jl")
include("interface.jl")
include("decompression.jl")
include("check.jl")
include("examples.jl")

export NaturalOrder, RandomOrder, LargestFirst
export DynamicDegreeBasedOrder, SmallestLast, IncidenceDegree, DynamicLargestFirst
export ColoringProblem, GreedyColoringAlgorithm, AbstractColoringResult
export coloring
export column_colors, row_colors
export column_groups, row_groups
export compress, decompress, decompress!, decompress_single_color!

end
