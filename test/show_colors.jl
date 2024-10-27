using SparseMatrixColorings
using SparseMatrixColorings: show_colors
using SparseArrays
using Colors
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

    img = show_colors(result)
    @test size(img) == size(S)
    @test img isa Matrix{<:Colorant}

    h, w = size(S)
    scale = 3
    img = show_colors(result; scale=scale)
    @test size(img) == (h * scale, w * scale)
    @test img isa Matrix{<:Colorant}

    pad = 2
    img = show_colors(result; scale=scale, pad=pad)
    @test size(img) == (h * (scale + pad) + pad, w * (scale + pad) + pad)
    @test img isa Matrix{<:Colorant}

    @testset "color cycling" begin
        colorscheme = [RGB(0, 0, 0), RGB(1, 1, 1)] # 2 colors, whereas S requires 3
        img = @test_logs (:warn,) show_colors(result; colorscheme=colorscheme)
        @test size(img) == size(S)
        @test img isa Matrix{<:Colorant}

        img = show_colors(result; colorscheme=colorscheme, warn=false)
        @test size(img) == size(S)
        @test img isa Matrix{<:Colorant}
    end
end

@testset "Errors" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:row)
    result = coloring(S, problem, algo)
    @testset "scale too small" begin
        @test_throws ArgumentError show_colors(result; scale=-24)
        @test_throws ArgumentError show_colors(result; scale=0)
    end
    @testset "pad too small" begin
        @test_throws ArgumentError show_colors(result; pad=-1)
        @test_nowarn show_colors(result; pad=0)
    end
    # @testset "Unsupported partitions" begin
    #     problem = ColoringProblem(; structure=:nonsymmetric, partition=:bidirectional) # TODO: not implemented by SMC
    #     result = coloring(S, problem, algo)
    #     @test_throws ErrorException show_colors(result)
    # end
end
