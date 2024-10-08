using SparseMatrixColorings
using SparseMatrixColorings: show_colors
using SparseArrays
using Images
using Test

S = sparse([
    0 0 1 1 0 1
    1 0 0 0 1 0
    0 1 0 0 1 0
    0 1 1 0 0 0
]);
algo = GreedyColoringAlgorithm(; decompression=:direct)

@testset "$partition" for partition in (:column, :row)
    problem = ColoringProblem(; structure=:nonsymmetric, partition=partition)
    result = coloring(S, problem, algo)

    @testset "Color scheme too small" begin
        img = show_colors(result)
        @test size(img) == size(S)
        @test img isa Matrix{<:Colorant}

        h, w = size(S)
        scale = 3
        img = show_colors(result; scale=scale)
        @test size(img) == (h * scale, w * scale)
        @test img isa Matrix{<:Colorant}

        padding = 2
        img = show_colors(result; scale=scale, padding=padding)
        @test size(img) ==
            (h * (scale + padding) - padding, w * (scale + padding) - padding)
        @test img isa Matrix{<:Colorant}
    end
end

@testset "Errors" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:row)
    result = coloring(S, problem, algo)
    @testset "scale too small" begin
        @test_throws ErrorException show_colors(result; scale=-24)
        @test_throws ErrorException show_colors(result; scale=0)
    end
    @testset "padding too small" begin
        @test_throws ErrorException show_colors(result; padding=-1)
        @test_nowarn show_colors(result; padding=0)
    end
    @testset "colorscheme too small" begin
        colorscheme = [RGB(0, 0, 0), RGB(1, 1, 1)] # 2 colors, whereas S requires 3
        @test_throws ErrorException show_colors(result; colorscheme=colorscheme)
    end
    # @testset "Unsupported partitions" begin
    #     problem = ColoringProblem(; structure=:nonsymmetric, partition=:bidirectional) # TODO: not implemented by SMC
    #     result = coloring(S, problem, algo)
    #     @test_throws ErrorException show_colors(result)
    # end
end
