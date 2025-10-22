using ADTypes: ADTypes
using SparseMatrixColorings
using SparseMatrixColorings: InvalidColoringError
using Test

matrix_template = ones(Bool, 10, 20)
sym_matrix_template = ones(Bool, 10, 10)

@testset "Column coloring" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)
    color = collect(1:20)
    algo = ConstantColoringAlgorithm(matrix_template, color; partition=:column)
    wrong_algo = ConstantColoringAlgorithm{:row}(matrix_template, color)
    wrong_color = ConstantColoringAlgorithm{:column}(matrix_template, ones(Int, 20))
    @test_throws DimensionMismatch coloring(transpose(matrix_template), problem, algo)
    @test_throws MethodError coloring(matrix_template, problem, wrong_algo)
    @test_throws InvalidColoringError coloring(matrix_template, problem, wrong_color)
    result = coloring(matrix_template, problem, algo)
    @test column_colors(result) == color
    @test ADTypes.column_coloring(matrix_template, algo) == color
    @test_throws MethodError ADTypes.row_coloring(matrix_template, algo)
end

@testset "Row coloring" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:row)
    color = collect(1:10)
    algo = ConstantColoringAlgorithm(matrix_template, color; partition=:row)
    wrong_algo = ConstantColoringAlgorithm{:column}(matrix_template, color)
    wrong_color = ConstantColoringAlgorithm{:row}(matrix_template, ones(Int, 10))
    @test_throws DimensionMismatch coloring(transpose(matrix_template), problem, algo)
    @test_throws MethodError coloring(matrix_template, problem, wrong_algo)
    @test_throws InvalidColoringError coloring(matrix_template, problem, wrong_color)
    result = coloring(matrix_template, problem, algo)
    @test row_colors(result) == color
    @test ADTypes.row_coloring(matrix_template, algo) == color
    @test_throws MethodError ADTypes.column_coloring(matrix_template, algo)
end

@testset "Symmetric coloring" begin
    problem = ColoringProblem(; structure=:symmetric, partition=:column)
    color = collect(1:10)
    algo = ConstantColoringAlgorithm(
        sym_matrix_template, color; partition=:column, structure=:symmetric
    )
    wrong_algo = ConstantColoringAlgorithm{:column,:nonsymmetric}(
        sym_matrix_template, color
    )
    wrong_color = ConstantColoringAlgorithm{:column,:symmetric}(
        sym_matrix_template, ones(Int, 20)
    )
    @test_throws DimensionMismatch coloring(matrix_template, problem, algo)
    @test_throws MethodError coloring(sym_matrix_template, problem, wrong_algo)
    @test_throws InvalidColoringError coloring(sym_matrix_template, problem, wrong_color)
    result = coloring(sym_matrix_template, problem, algo)
    @test column_colors(result) == color
    @test ADTypes.symmetric_coloring(sym_matrix_template, algo) == color
    @test_throws MethodError ADTypes.column_coloring(sym_matrix_template, algo)
end
