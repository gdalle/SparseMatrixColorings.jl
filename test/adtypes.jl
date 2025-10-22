using ADTypes: ADTypes
using SparseArrays
using LinearAlgebra
using SparseMatrixColorings
using Test

@testset "NoColoringAlgorithm" begin
    @testset "Column coloring" begin
        problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)
        algo = ADTypes.NoColoringAlgorithm()
        A = sprand(10, 20, 0.3)
        result = coloring(A, problem, algo)
        B = compress(A, result)
        @test size(B) == size(A)
        @test column_colors(result) == ADTypes.column_coloring(A, algo)
        @test decompress(B, result) == A
    end

    @testset "Row coloring" begin
        problem = ColoringProblem(; structure=:nonsymmetric, partition=:row)
        algo = ADTypes.NoColoringAlgorithm()
        A = sprand(10, 20, 0.3)
        result = coloring(A, problem, algo)
        B = compress(A, result)
        @test size(B) == size(A)
        @test row_colors(result) == ADTypes.row_coloring(A, algo)
        @test decompress(B, result) == A
    end

    @testset "Symmetric coloring" begin
        problem = ColoringProblem(; structure=:symmetric, partition=:column)
        algo = ADTypes.NoColoringAlgorithm()
        A = Symmetric(sprand(20, 20, 0.3))
        result = coloring(A, problem, algo)
        B = compress(A, result)
        @test size(B) == size(A)
        @test column_colors(result) == ADTypes.column_coloring(A, algo)
        @test decompress(B, result) == A
    end

    @testset "Bicoloring" begin
        problem = ColoringProblem(; structure=:nonsymmetric, partition=:bidirectional)
        algo = ADTypes.NoColoringAlgorithm()
        A = sprand(10, 20, 0.3)
        result = coloring(A, problem, algo)
        Br, Bc = compress(A, result)
        @test decompress(Br, Bc, result) == A
    end
end
