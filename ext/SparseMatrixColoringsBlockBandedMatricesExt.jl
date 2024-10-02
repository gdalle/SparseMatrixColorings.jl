module SparseMatrixColoringsBlockBandedMatricesExt

if isdefined(Base, :get_extension)
    using BlockArrays: blockaxes, blockfirsts, blocklasts, blocksize, blocklengths
    using BlockBandedMatrices:
        BlockBandedMatrix,
        blockbandrange,
        blockbandwidths,
        blocklengths,
        blocksize,
        subblockbandwidths
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
    using ..BlockArrays: blockaxes, blockfirsts, blocklasts, blocksize, blocklengths
    using ..BlockBandedMatrices:
        BlockBandedMatrix,
        blockbandrange,
        blockbandwidths,
        blocklengths,
        blocksize,
        subblockbandwidths
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
This code is partly taken from ArrayInterface.jl and FiniteDiff.jl
https://github.com/JuliaArrays/ArrayInterface.jl
https://github.com/JuliaDiff/FiniteDiff.jl
=#

function SMC.coloring(
    A::BlockBandedMatrix,
    ::ColoringProblem{:nonsymmetric,:column},
    algo::GreedyColoringAlgorithm;
    kwargs...,
)
    # consider blocks of columns
    nb_blocks = blocksize(A, 2)
    nb_cols_in_block = blocklengths(axes(A, 2))
    first_col_in_block = blockfirsts(axes(A, 2))
    last_col_in_block = blocklasts(axes(A, 2))

    # give a macroscopic color to each block, so that 2 blocks of columns with the same macro color do not intersect
    # same idea as for BandedMatrices
    nb_macrocolors = length(blockbandrange(A))
    macrocolor = cycle_range(nb_macrocolors, nb_blocks)

    # for each macroscopic color, count how many microscopic colors will be needed
    # columns within a block are colored naively with all distinct micro colors
    nb_colors_in_macrocolor = [
        maximum(nb_cols_in_block[mc:nb_macrocolors:nb_blocks]; init=0) for
        mc in 1:nb_macrocolors
    ]
    color_shift_in_macrocolor = vcat(0, cumsum(nb_colors_in_macrocolor)[1:(end - 1)])

    # assign a microscopic color to each column as a function of its macroscopic color and its position within the block
    color = Vector{Int}(undef, size(A, 2))
    for b in 1:nb_blocks
        mc = macrocolor[b]
        shift = color_shift_in_macrocolor[mc]
        for j in first_col_in_block[b]:last_col_in_block[b]
            color[j] = shift + (j - first_col_in_block[b] + 1)
        end
    end

    bg = BipartiteGraph(A)
    return ColumnColoringResult(A, bg, color)
end

end
