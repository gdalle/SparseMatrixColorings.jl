using CSV
using DataFrames
using LinearAlgebra
using SparseArray
using SparseMatrixColorings
using Test

## Banded

#=
Per the recovery paper "Efficient computation of sparse Hessians using Coloring and Automatic Differentiation", for a symmetric banded matrix with ρ non-diagonal bands:
- acyclic coloring uses ⌊ρ/2⌋+1 colors
- star coloring uses 2⌊ρ/2⌋+1 colors
=#

problem = ColoringProblem(; structure=:symmetric, partition=:column)

function banded_matrix(n::Integer, ρ::Integer)
    return spdiagm([k => ones(Bool, n - abs(k)) for k in (-(ρ ÷ 2)):(ρ ÷ 2)]...)
end

@testset "Star coloring" begin
    algo = GreedyColoringAlgorithm(; decompression=:direct)
    for n in [5, 10, 20] .* 1000
        S = banded_matrix(n, 10)
        direct_result = coloring(S, problem, algo)
        @test length(column_groups(direct_result)) == 11

        S = banded_matrix(n, 20)
        direct_result = coloring(S, problem, algo)
        @test length(column_groups(direct_result)) == 21
    end
end

@testset "Acyclic coloring" begin
    algo = GreedyColoringAlgorithm(; decompression=:substitution)
    for n in [5, 10, 20] .* 1000
        S = banded_matrix(n, 10)
        substitution_result = coloring(S, problem, algo)
        @test length(column_groups(substitution_result)) == 6

        S = banded_matrix(n, 20)
        substitution_result = coloring(S, problem, algo)
        @test length(column_groups(substitution_result)) == 11
    end
end
