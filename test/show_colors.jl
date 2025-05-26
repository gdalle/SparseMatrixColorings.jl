using SparseMatrixColorings
using SparseMatrixColorings: show_colors
using SparseArrays
using Colors
using Test

A = sparse([
    0 0 1 1 0 1
    1 0 0 0 1 0
    0 1 0 0 1 0
    0 1 1 0 0 0
]);
algo = GreedyColoringAlgorithm(; decompression=:direct)

@testset "$partition" for partition in (:column, :row, :bidirectional)
    problem = ColoringProblem(; structure=:nonsymmetric, partition=partition)
    result = coloring(A, problem, algo)

    if partition != :bidirectional
        B = compress(A, result)

        A_img, B_img = show_colors(result)
        @test size(A_img) == size(A)
        @test size(B_img) == size(B)
        @test A_img isa Matrix{<:Colorant}
        @test B_img isa Matrix{<:Colorant}

        scale = 3
        A_img, B_img = show_colors(result; scale=scale)
        @test size(A_img) == size(A) .* scale
        @test size(B_img) == size(B) .* scale
        @test A_img isa Matrix{<:Colorant}

        pad = 2
        border = 3
        A_img, B_img = show_colors(result; scale=scale, border=border, pad=pad)
        @test size(A_img) == size(A) .* (scale + 2border + pad) .+ pad
        @test size(B_img) == size(B) .* (scale + 2border + pad) .+ pad
        @test A_img isa Matrix{<:Colorant}

        @testset "color cycling" begin
            colorscheme = [RGB(0, 0, 0), RGB(1, 1, 1)] # 2 colors, whereas S requires 3
            A_img, _ = @test_logs (:warn,) show_colors(result; colorscheme)
            @test size(A_img) == size(A)
            @test A_img isa Matrix{<:Colorant}

            A_img, _ = show_colors(result; colorscheme, warn=false)
            @test size(A_img) == size(A)
            @test A_img isa Matrix{<:Colorant}
        end
    else
        Br, Bc = compress(A, result)

        scale = 3
        Arc_img, Ar_img, Ac_img, Br_img, Bc_img = show_colors(result; scale=scale)
        @test size(Arc_img) == size(A) .* scale
        @test size(Ar_img) == size(A) .* scale
        @test size(Ac_img) == size(A) .* scale
        @test size(Br_img) == size(Br) .* scale
        @test size(Bc_img) == size(Bc) .* scale
        @test Arc_img isa Matrix{<:Colorant}
        @test Ar_img isa Matrix{<:Colorant}
        @test Ac_img isa Matrix{<:Colorant}
        @test Br_img isa Matrix{<:Colorant}
        @test Bc_img isa Matrix{<:Colorant}
    end
end

@testset "Errors" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:row)
    result = coloring(A, problem, algo)
    @testset "scale too small" begin
        @test_throws ArgumentError show_colors(result; scale=-24)
        @test_throws ArgumentError show_colors(result; scale=0)
    end
    @testset "pad too small" begin
        @test_throws ArgumentError show_colors(result; pad=-1)
        @test_nowarn show_colors(result; pad=0)
    end
    @testset "border too small" begin
        @test_throws ArgumentError show_colors(result; border=-1)
        @test_nowarn show_colors(result; border=0)
    end
end
