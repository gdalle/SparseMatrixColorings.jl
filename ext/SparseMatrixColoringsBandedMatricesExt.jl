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

function SMC.decompress!(A::BandedMatrix, B::AbstractMatrix, result::ColumnColoringResult)
    color = column_colors(result)
    for j in axes(A, 2)
        c = color[j]
        for i in colrange(A, j)
            A[i, j] = B[i, c]
        end
    end
    return A
end

function SMC.decompress!(A::BandedMatrix, B::AbstractMatrix, result::RowColoringResult)
    color = row_colors(result)
    for i in axes(A, 1)
        c = color[i]
        for j in rowrange(A, i)
            A[i, j] = B[c, j]
        end
    end
    return A
end

end
