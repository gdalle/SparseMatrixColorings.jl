using SparseMatrixColorings
using Test

@test ColoringProblem{:nonsymmetric,:column}() == ColoringProblem()
@test ColoringProblem{:symmetric,:column}() ==
    ColoringProblem(; structure=:symmetric, partition=:column)

@test_throws ArgumentError ColoringProblem(; structure=:weird, partition=:column)
@test_throws ArgumentError ColoringProblem(; structure=:symmetric, partition=:row)

@test GreedyColoringAlgorithm{:direct}() == GreedyColoringAlgorithm()
@test GreedyColoringAlgorithm{:substitution}() ==
    GreedyColoringAlgorithm(; decompression=:substitution)

@test_throws ArgumentError GreedyColoringAlgorithm(decompression=:weird)
