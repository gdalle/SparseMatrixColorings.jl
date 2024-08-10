using ADTypes: column_coloring, row_coloring, symmetric_coloring
using Base.Iterators: product
using Compat
using LinearAlgebra
using SparseArrays
using SparseMatrixColorings
using SparseMatrixColorings:
    DefaultColoringResult,
    group_by_color,
    structurally_orthogonal_columns,
    symmetrically_orthogonal_columns,
    directly_recoverable_columns,
    matrix_versions,
    respectful_similar,
    what_fig_41,
    what_fig_61,
    efficient_fig_1,
    efficient_fig_4
using Test

algo = GreedyColoringAlgorithm()

@testset "Column decompression" begin
    A0 = sparse([
        1 0 2
        0 3 4
        5 0 0
    ])
    B0 = [
        1 2
        3 4
        5 0
    ]
    color0 = [1, 1, 2]
    result0 = DefaultColoringResult{:nonsymmetric,:column,:direct}(A0, color0)
    @test structurally_orthogonal_columns(A0, color0)
    @test directly_recoverable_columns(A0, color0)
    @test decompress(B0, result0) == A0
    for A in matrix_versions(A0)
        @test decompress!(respectful_similar(A), B0, result0) == A
    end
end;

@testset "Row decompression" begin
    A0 = sparse([
        1 0 3
        0 2 0
        4 5 0
    ])
    B0 = [
        1 2 3
        4 5 0
    ]
    color0 = [1, 1, 2]
    result0 = DefaultColoringResult{:nonsymmetric,:row,:direct}(A0, color0)
    @test structurally_orthogonal_columns(transpose(A0), color0)
    @test directly_recoverable_columns(transpose(A0), color0)
    @test decompress(B0, result0) == A0
    for A in matrix_versions(A0)
        @test decompress!(respectful_similar(A), B0, result0) == A
    end
end;

@testset "Symmetric decompression" begin
    @testset "Direct - Fig 4.1 from 'What color is your Jacobian'" begin
        example = what_fig_41()
        A0, B0, color0 = example.A, example.B, example.color
        result0 = DefaultColoringResult{:symmetric,:column,:direct}(A0, color0)
        @test symmetrically_orthogonal_columns(A0, color0)
        @test directly_recoverable_columns(A0, color0)
        @test decompress(B0, result0) == A0
        for A in matrix_versions(A0)
            @test decompress!(respectful_similar(A), B0, result0) == A
        end
    end

    @testset "Substitution - Fig 6.1 from 'What color is your Jacobian'" begin
        example = what_fig_61()
        A0, B0, color0 = example.A, example.B, example.color
        result0 = DefaultColoringResult{:symmetric,:column,:substitution}(A0, color0)
        result = coloring(
            A0,
            ColoringProblem(;
                structure=:symmetric, partition=:column, decompression=:substitution
            ),
            GreedyColoringAlgorithm(),
        )  # returns a TreeSetColoringResult
        group = column_groups(result)
        B = stack(group; dims=2) do g
            dropdims(sum(A0[:, g]; dims=2); dims=2)
        end
        @test column_colors(result) != color0
        @test B != B0
        @test decompress(B, result) ≈ A0
        @test decompress(B0, result0) ≈ A0
        for A in matrix_versions(A0)
            @test decompress!(respectful_similar(A), B0, result0) ≈ A
            @test decompress!(respectful_similar(A), B, result) ≈ A
        end
    end

    @testset "Direct - Fig 1 from 'Efficient computation of sparse hessians using coloring and AD'" begin
        example = efficient_fig_1()
        A0, B0, color0 = example.A, example.B, example.color
        result0 = DefaultColoringResult{:symmetric,:column,:direct}(A0, color0)
        @test symmetrically_orthogonal_columns(A0, color0)
        @test directly_recoverable_columns(A0, color0)
        @test decompress(B0, result0) == A0
        for A in matrix_versions(A0)
            @test decompress!(respectful_similar(A), B0, result0) == A
        end
    end

    @testset "Substitution - Fig 4 from 'Efficient computation of sparse hessians using coloring and AD'" begin
        example = efficient_fig_4()
        A0, B0, color0 = example.A, example.B, example.color
        result0 = DefaultColoringResult{:symmetric,:column,:substitution}(A0, color0)
        result = coloring(
            A0,
            ColoringProblem(;
                structure=:symmetric, partition=:column, decompression=:substitution
            ),
            GreedyColoringAlgorithm(),
        )  # returns a TreeSetColoringResult
        @test column_colors(result) == color0
        @test decompress(B0, result0) ≈ A0
        @test decompress(B0, result) ≈ A0
        for A in matrix_versions(A0)
            @test decompress!(respectful_similar(A), B0, result0) ≈ A
            @test decompress!(respectful_similar(A), B0, result) ≈ A
        end
    end
end;
