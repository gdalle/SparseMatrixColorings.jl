using ADTypes: ADTypes
using SparseMatrixColorings
using StableRNGs
using Test

rng = StableRNG(63)

A = sprand(rng, 100, 200, 0.05)

@testset "Column coloring" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)
    color = fast_coloring(A, problem, GreedyColoringAlgorithm())
    algo = ConstantColoringAlgorithm(A, color; partition=:column)
    algo_tolerant = ConstantColoringAlgorithm(
        A, color; partition=:column, allow_denser=true
    )
    wrong_algo = ConstantColoringAlgorithm(A, color; partition=:row)
    @test_throws DimensionMismatch coloring(transpose(A), problem, algo)
    @test_throws MethodError coloring(A, problem, wrong_algo)
    result = coloring(A, problem, algo)
    result_tolerant = coloring(A, problem, algo_tolerant)
    B = compress(A, result)
    @test column_colors(result) == color
    @test ADTypes.column_coloring(A, algo) == color
    @test_throws MethodError ADTypes.row_coloring(A, algo)
    A2 = copy(A)
    for added_coeff in 1:10
        i, j = rand(rng, axes(A, 1)), rand(rng, axes(A, 2))
        A2[i, j] = 1
    end
    @test_throws DimensionMismatch decompress!(A2, B, result)
    @test decompress!(A2, B, result_tolerant) == A
end

@testset "Row coloring" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:row)
    color = fast_coloring(A, problem, GreedyColoringAlgorithm())
    algo = ConstantColoringAlgorithm(A, color; partition=:row)
    algo_tolerant = ConstantColoringAlgorithm(A, color; partition=:row, allow_denser=true)
    @test_throws DimensionMismatch coloring(transpose(A), problem, algo)
    result = coloring(A, problem, algo)
    result_tolerant = coloring(A, problem, algo_tolerant)
    B = compress(A, result)
    @test row_colors(result) == color
    @test ADTypes.row_coloring(A, algo) == color
    @test_throws MethodError ADTypes.column_coloring(A, algo)
    A2 = copy(A)
    for added_coeff in 1:10
        i, j = rand(rng, axes(A, 1)), rand(rng, axes(A, 2))
        A2[i, j] = 1
    end
    @test decompress!(A2, B, result_tolerant) == A
    @test_throws DimensionMismatch decompress!(A2, B, result)
end

@testset "Symmetric coloring" begin
    wrong_problem = ColoringProblem(; structure=:symmetric, partition=:column)
    color = ones(Int, size(A, 2))
    algo = ConstantColoringAlgorithm(A, color; partition=:column)
    @test_throws MethodError coloring(A, wrong_problem, algo)
    @test_throws MethodError ADTypes.symmetric_coloring(A, algo)
end
