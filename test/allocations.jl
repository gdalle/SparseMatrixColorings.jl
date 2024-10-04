using Chairmarks
using LinearAlgebra
using SparseArrays
using SparseMatrixColorings
using SparseMatrixColorings: BipartiteGraph, partial_distance2_coloring!
using StableRNGs
using Test

rng = StableRNG(63)

function test_noallocs_distance2_coloring(n)
    bench = @be (;
        bg=BipartiteGraph(sprand(rng, n, n, 5 / n)),
        color=Vector{Int}(undef, n),
        forbidden_colors=Vector{Int}(undef, n),
    ) partial_distance2_coloring!(_.color, _.forbidden_colors, _.bg, Val(1), 1:n) evals = 1
    @test minimum(bench).allocs == 0
end

@testset "Distance-2 coloring" begin
    test_noallocs_distance2_coloring(1000)
end;

function test_noallocs_sparse_decompression(
    n::Integer; structure::Symbol, partition::Symbol, decompression::Symbol
)
    A = sparse(Symmetric(sprand(rng, n, n, 5 / n)))
    result = coloring(
        A, ColoringProblem(; structure, partition), GreedyColoringAlgorithm(; decompression)
    )
    B = compress(A, result)

    @testset "Full decompression" begin
        bench1_full = @be similar(A) decompress!(_, B, result) evals = 1
        bench2_full = @be similar(Matrix(A)) decompress!(_, B, result) evals = 1
        @test minimum(bench1_full).allocs == 0
        @test minimum(bench2_full).allocs == 0
    end
    @testset "Single-color decompression" begin
        if decompression == :direct
            b = if partition == :column
                B[:, 1]
            else
                B[1, :]
            end
            bench1_singlecolor = @be similar(A) decompress_single_color!(_, b, 1, result) evals =
                1
            bench2_singlecolor = @be similar(Matrix(A)) decompress_single_color!(
                _, b, 1, result
            ) evals = 1
            @test minimum(bench1_singlecolor).allocs == 0
            @test minimum(bench2_singlecolor).allocs == 0
        end
    end
    @testset "Triangle decompression" begin
        if structure == :symmetric
            bench1_triangle = @be similar(triu(A)) decompress!(_, B, result, :U) evals = 1
            bench2_triangle = @be similar(Matrix(A)) decompress!(_, B, result, :U) evals = 1
            @test minimum(bench1_triangle).allocs == 0
            @test minimum(bench2_triangle).allocs == 0
        end
    end
    @testset "Single-color triangle decompression" begin
        if structure == :symmetric && decompression == :direct
            b = B[:, 1]
            bench1_singlecolor_triangle = @be similar(triu(A)) decompress_single_color!(
                _, b, 1, result, :U
            ) evals = 1
            bench2_singlecolor_triangle = @be similar(Matrix(A)) decompress_single_color!(
                _, b, 1, result, :U
            ) evals = 1
            @test minimum(bench1_singlecolor_triangle).allocs == 0
            @test minimum(bench2_singlecolor_triangle).allocs == 0
        end
    end
end

function test_noallocs_structured_decompression(
    n::Integer; structure::Symbol, partition::Symbol, decompression::Symbol
)
    @testset "$(nameof(typeof(A)))" for A in [
        Diagonal(rand(n)),
        Bidiagonal(rand(n), rand(n - 1), 'U'),
        Bidiagonal(rand(n), rand(n - 1), 'L'),
        Tridiagonal(rand(n - 1), rand(n), rand(n - 1)),
    ]
        result = coloring(
            A,
            ColoringProblem(; structure, partition),
            GreedyColoringAlgorithm(; decompression),
        )
        B = compress(A, result)
        bench = @be similar(A) decompress!(_, B, result) evals = 1
        @test minimum(bench).allocs == 0
    end
end

@testset "Sparse decompression" begin
    @testset "$structure - $partition - $decompression" for (
        structure, partition, decompression
    ) in [
        (:nonsymmetric, :column, :direct),
        (:nonsymmetric, :row, :direct),
        (:symmetric, :column, :direct),
        (:symmetric, :column, :substitution),
    ]
        test_noallocs_sparse_decompression(1000; structure, partition, decompression)
    end
end;

@testset "Structured decompression" begin
    @testset "$structure - $partition - $decompression" for (
        structure, partition, decompression
    ) in [
        (:nonsymmetric, :column, :direct), (:nonsymmetric, :row, :direct)
    ]
        test_noallocs_structured_decompression(1000; structure, partition, decompression)
    end
end;
