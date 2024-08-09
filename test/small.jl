using Base.Iterators: product
using Compat
using LinearAlgebra
using SparseArrays
using SparseMatrixColorings
using SparseMatrixColorings:
    DefaultColoringResult,
    decompress,
    decompress!,
    group_by_color,
    structurally_orthogonal_columns,
    symmetrically_orthogonal_columns,
    directly_recoverable_columns,
    matrix_versions,
    respectful_similar,
    what_fig_41,
    efficient_fig_1
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
    @testset "Fig 4.1 from 'What color is your Jacobian'" begin
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

    @testset "Fig 1 from 'Efficient computation of sparse hessians using coloring and AD'" begin
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
end;
