using LinearAlgebra
using SparseMatrixColorings
using Test

column_problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)
row_problem = ColoringProblem(; structure=:nonsymmetric, partition=:row)

algo = GreedyColoringAlgorithm()

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
