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
    Bidiagonal,
    Diagonal,
    Hermitian,
    LowerTriangular,
    Symmetric,
    Transpose,
    Tridiagonal,
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
include("constant.jl")
include("adtypes.jl")
include("decompression.jl")
include("structured.jl")
include("check.jl")
include("examples.jl")

export NaturalOrder, RandomOrder, LargestFirst
export ColoringProblem, GreedyColoringAlgorithm, AbstractColoringResult
export ConstantColoringAlgorithm
export coloring
export column_colors, row_colors
export column_groups, row_groups
export sparsity_pattern
export compress, decompress, decompress!, decompress_single_color!

if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require BandedMatrices = "aae01518-5342-5314-be14-df237901396f" include(
            "../ext/SparseMatrixColoringsBandedMatricesExt.jl"
        )
        @require BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0" include(
            "../ext/SparseMatrixColoringsBlockBandedMatricesExt.jl"
        )
    end
end

end
