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
include("constant.jl")
include("adtypes.jl")
include("decompression.jl")
include("check.jl")
include("examples.jl")
include("images.jl")

export NaturalOrder, RandomOrder, LargestFirst
export ColoringProblem, GreedyColoringAlgorithm, AbstractColoringResult
export ConstantColoringAlgorithm
export coloring
export column_colors, row_colors
export column_groups, row_groups
export sparsity_pattern
export compress, decompress, decompress!, decompress_single_color!

function __init__()
    @static if !isdefined(Base, :get_extension)
        @require Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0" include(
            "../ext/SparseMatrixColoringsImagesExt.jl"
        )
    end
end

end
