module SparseMatrixColoringsBlockBandedMatricesExt

if isdefined(Base, :get_extension)
    using BlockArrays: blockaxes, blockfirsts, blocklasts, blocksize, blocklengths
    using BlockBandedMatrices:
        BandedBlockBandedMatrix,
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
        BandedBlockBandedMatrix,
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

function subblockbandrange(A::BandedBlockBandedMatrix)
    u, l = subblockbandwidths(A)
    return (-l):u
end

function blockbanded_coloring(
    A::Union{BlockBandedMatrix,BandedBlockBandedMatrix}, dim::Integer
)
    # consider blocks of columns or rows (let's call them vertices) depending on `dim`
    nb_blocks = blocksize(A, dim)
    nb_in_block = blocklengths(axes(A, dim))
    first_in_block = blockfirsts(axes(A, dim))
    last_in_block = blocklasts(axes(A, dim))
    color = zeros(Int, size(A, dim))

    # give a macroscopic color to each block, so that 2 blocks with the same macro color are orthogonal
    # same idea as for BandedMatrices
    nb_macrocolors = length(blockbandrange(A))
    macrocolor = cycle_range(nb_macrocolors, nb_blocks)

    width = if A isa BandedBlockBandedMatrix
        # vertices within a block are colored cleverly using bands
        length(subblockbandrange(A))
    else
        # vertices within a block are colored naively with distinct micro colors (~ infinite band width)
        minimum(size(A))
    end

    # for each macroscopic color, count how many microscopic colors will be needed
    nb_colors_in_macrocolor = zeros(Int, nb_macrocolors)
    for mc in 1:nb_macrocolors
        largest_nb_in_macrocolor = maximum(nb_in_block[mc:nb_macrocolors:nb_blocks]; init=0)
        nb_colors_in_macrocolor[mc] = min(width, largest_nb_in_macrocolor)
    end
    color_shift_in_macrocolor = vcat(0, cumsum(nb_colors_in_macrocolor)[1:(end - 1)])

    # assign a microscopic color to each column as a function of its macroscopic color and its position within the block
    for b in 1:nb_blocks
        block_color = cycle_range(width, nb_in_block[b])
        shift = color_shift_in_macrocolor[macrocolor[b]]
        color[first_in_block[b]:last_in_block[b]] .= shift .+ block_color
    end

    return color
end

function SMC.coloring(
    A::Union{BlockBandedMatrix,BandedBlockBandedMatrix},
    ::ColoringProblem{:nonsymmetric,:column},
    algo::GreedyColoringAlgorithm;
    kwargs...,
)
    color = blockbanded_coloring(A, 2)
    bg = BipartiteGraph(A)
    return ColumnColoringResult(A, bg, color)
end

function SMC.coloring(
    A::Union{BlockBandedMatrix,BandedBlockBandedMatrix},
    ::ColoringProblem{:nonsymmetric,:row},
    algo::GreedyColoringAlgorithm;
    kwargs...,
)
    color = blockbanded_coloring(A, 1)
    bg = BipartiteGraph(A)
    return RowColoringResult(A, bg, color)
end

end
