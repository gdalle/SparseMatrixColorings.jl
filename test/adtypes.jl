using ADTypes: ADTypes
using SparseArrays
using SparseMatrixColorings
using Test

@testset "Column coloring" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)
    algo = ADTypes.NoColoringAlgorithm()
    A = sprand(10, 20, 0.1)
    result = coloring(A, problem, algo)
    B = compress(A, result)
    @test size(B) == size(A)
    @test column_colors(result) == ADTypes.column_coloring(A, algo)
    @test decompress(B, result) == A
end

@testset "Row coloring" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:row)
    algo = ADTypes.NoColoringAlgorithm()
    A = sprand(10, 20, 0.1)
    result = coloring(A, problem, algo)
    B = compress(A, result)
    @test size(B) == size(A)
    @test row_colors(result) == ADTypes.row_coloring(A, algo)
    @test decompress(B, result) == A
end
