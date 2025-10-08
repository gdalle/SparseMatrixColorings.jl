using ADTypes: ADTypes
using SparseMatrixColorings
using Test

matrix_template = ones(100, 200)

@testset "Column coloring" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)
    color = rand(1:5, size(matrix_template, 2))
    algo = ConstantColoringAlgorithm(matrix_template, color; partition=:column)
    wrong_algo = ConstantColoringAlgorithm(matrix_template, color; partition=:row)
    @test_throws DimensionMismatch coloring(transpose(matrix_template), problem, algo)
    @test_throws MethodError coloring(matrix_template, problem, wrong_algo)
    result = coloring(matrix_template, problem, algo)
    @test column_colors(result) == color
    @test ADTypes.column_coloring(matrix_template, algo) == color
    @test_throws MethodError ADTypes.row_coloring(matrix_template, algo)
end

@testset "Row coloring" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:row)
    color = rand(1:5, size(matrix_template, 1))
    algo = ConstantColoringAlgorithm(matrix_template, color; partition=:row)
    @test_throws DimensionMismatch coloring(transpose(matrix_template), problem, algo)
    result = coloring(matrix_template, problem, algo)
    @test row_colors(result) == color
    @test ADTypes.row_coloring(matrix_template, algo) == color
    @test_throws MethodError ADTypes.column_coloring(matrix_template, algo)
end
