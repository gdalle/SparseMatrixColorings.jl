"""
    SparseMatrixColorings

$README

## Exports

$EXPORTS
"""
module SparseMatrixColorings

using DataStructures: DisjointSets, find_root!, root_union!, num_groups
using ADTypes: ADTypes
using Compat: @compat, stack
using DocStringExtensions: README, EXPORTS, SIGNATURES, TYPEDEF, TYPEDFIELDS
using LinearAlgebra:
    Adjoint,
    Diagonal,
    Symmetric,
    Transpose,
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
include("examples.jl")

export ColoringProblem, GreedyColoringAlgorithm, AbstractColoringResult
export coloring
export column_colors, row_colors
export column_groups, row_groups
export compress, decompress, decompress!

@compat public NaturalOrder, RandomOrder, LargestFirst
@compat public DynamicDegreeBasedOrder, SmallestLast, IncidenceDegree, DynamicLargestFirst

end
