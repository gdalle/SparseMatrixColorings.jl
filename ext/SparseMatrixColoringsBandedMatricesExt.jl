module SparseMatrixColoringsBandedMatricesExt

using BandedMatrices: BandedMatrix, bandrange, bandwidths, colrange, rowrange
using SparseMatrixColorings:
    BipartiteGraph,
    ColoringProblem,
    ColumnColoringResult,
    StructuredColoringAlgorithm,
    RowColoringResult,
    column_colors,
    cycle_range,
    row_colors
import SparseMatrixColorings as SMC

#=
This code is partly taken from ArrayInterface.jl and FiniteDiff.jl
https://github.com/JuliaArrays/ArrayInterface.jl
https://github.com/JuliaDiff/FiniteDiff.jl
=#

function SMC.coloring(
    A::BandedMatrix,
    ::ColoringProblem{:nonsymmetric,:column},
    ::StructuredColoringAlgorithm;
    kwargs...,
)
    width = length(bandrange(A))
    color = cycle_range(width, size(A, 2))
    bg = BipartiteGraph(A)
    return ColumnColoringResult(A, bg, color)
end

function SMC.coloring(
    A::BandedMatrix,
    ::ColoringProblem{:nonsymmetric,:row},
    ::StructuredColoringAlgorithm;
    kwargs...,
)
    width = length(bandrange(A))
    color = cycle_range(width, size(A, 1))
    bg = BipartiteGraph(A)
    return RowColoringResult(A, bg, color)
end

end
