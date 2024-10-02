using LinearAlgebra
using SparseMatrixColorings
using SparseMatrixColorings: cycle_range
using Test

column_problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)
row_problem = ColoringProblem(; structure=:nonsymmetric, partition=:row)

algo = GreedyColoringAlgorithm()

@testset "Utils" begin
    @test cycle_range(2, 3) == [1, 2, 1]
    @test cycle_range(2, 4) == [1, 2, 1, 2]
    @test cycle_range(2, 5) == [1, 2, 1, 2, 1]
    @test cycle_range(3, 5) == [1, 2, 3, 1, 2]
    @test cycle_range(3, 6) == [1, 2, 3, 1, 2, 3]
    @test cycle_range(2, 1) == [1]
    @test cycle_range(3, 1) == [1]
    @test cycle_range(3, 2) == [1, 2]
end

@testset "Diagonal" begin
    for n in (1, 2, 10, 100, 1000)
        A = Diagonal(rand(n))
        # column
        result = coloring(A, column_problem, algo)
        B = compress(A, result)
        @test size(B, 2) == 1
        @test decompress(B, result) == A
        # row
        result = coloring(A, row_problem, algo)
        B = compress(A, result)
        @test size(B, 1) == 1
        @test decompress(B, result) == A
    end
end

@testset "Bidiagonal" begin
    for n in (2, 10, 100, 1000)
        A1 = Bidiagonal(rand(n), rand(n - 1), :U)
        A2 = Bidiagonal(rand(n), rand(n - 1), :L)
        for A in (A1, A2)
            # column
            result = coloring(A, column_problem, algo)
            B = compress(A, result)
            @test size(B, 2) == 2
            @test decompress(B, result) == A
            # row
            result = coloring(A, row_problem, algo)
            B = compress(A, result)
            @test size(B, 1) == 2
            @test decompress(B, result) == A
        end
    end
end

@testset "Tridiagonal" begin
    for n in (2, 10, 100, 1000)
        A1 = Tridiagonal(rand(n - 1), rand(n), rand(n - 1))
        A2 = Tridiagonal(rand(n - 1), rand(n), rand(n - 1))
        for A in (A1, A2)
            # column
            result = coloring(A, column_problem, algo)
            B = compress(A, result)
            @test size(B, 2) == min(n, 3)
            @test decompress(B, result) == A
            # row
            result = coloring(A, row_problem, algo)
            B = compress(A, result)
            @test size(B, 1) == min(n, 3)
            @test decompress(B, result) == A
        end
    end
end
