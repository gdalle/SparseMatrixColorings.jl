module SparseMatrixColoringsBandedMatricesExt

if isdefined(Base, :get_extension)
    using BandedMatrices: BandedMatrix, bandwidths, bandrange
    using SparseMatrixColorings:
        BipartiteGraph,
        ColoringProblem,
        ColumnColoringResult,
        GreedyColoringAlgorithm,
        RowColoringResult,
        column_colors,
        cycle_range,
        row_colors
    import SparseMatrixColorings as SMC
else
    using ..BandedMatrices: BandedMatrix, bandwidths, bandrange
    using ..SparseMatrixColorings:
        BipartiteGraph,
        ColoringProblem,
        ColumnColoringResult,
        GreedyColoringAlgorithm,
        RowColoringResult,
        column_colors,
        cycle_range,
        row_colors
    import ..SparseMatrixColorings as SMC
end

#=
This code is partially taken from ArrayInterface.jl and FiniteDiff.jl
https://github.com/JuliaArrays/ArrayInterface.jl
https://github.com/JuliaDiff/FiniteDiff.jl
=#

function SMC.coloring(
    A::BandedMatrix,
    ::ColoringProblem{:nonsymmetric,:column},
    algo::GreedyColoringAlgorithm;
    kwargs...,
)
    l, u = bandwidths(A)
    width = u + l + 1
    color = cycle_range(width, size(A, 2))
    bg = BipartiteGraph(A)
    return ColumnColoringResult(A, bg, color)
end

function SMC.coloring(
    A::BandedMatrix,
    ::ColoringProblem{:nonsymmetric,:row},
    algo::GreedyColoringAlgorithm;
    kwargs...,
)
    l, u = bandwidths(A)
    width = u + l + 1
    color = cycle_range(width, size(A, 1))
    bg = BipartiteGraph(A)
    return RowColoringResult(A, bg, color)
end

function SMC.decompress!(A::BandedMatrix, B::AbstractMatrix, result::ColumnColoringResult)
    color = column_colors(result)
    m, n = size(A)
    l, u = bandwidths(A)
    for j in axes(A, 2)
        c = color[j]
        for i in max(1, j - u):min(m, j + l)
            A[i, j] = B[i, c]
        end
    end
    return A
end

function SMC.decompress!(A::BandedMatrix, B::AbstractMatrix, result::RowColoringResult)
    color = row_colors(result)
    m, n = size(A)
    l, u = bandwidths(A)
    for i in axes(A, 1)
        c = color[i]
        for j in max(1, i - l):min(n, i + u)
            A[i, j] = B[c, j]
        end
    end
    return A
end

end
