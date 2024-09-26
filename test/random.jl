using ADTypes: column_coloring, row_coloring, symmetric_coloring
using LinearAlgebra
using SparseArrays
using SparseMatrixColorings
using SparseMatrixColorings:
    structurally_orthogonal_columns,
    symmetrically_orthogonal_columns,
    directly_recoverable_columns
using StableRNGs
using Test

rng = StableRNG(63)

asymmetric_params = vcat(
    [(10, 20, p) for p in (0.0:0.1:0.5)],
    [(20, 10, p) for p in (0.0:0.1:0.5)],
    [(100, 200, p) for p in (0.01:0.01:0.05)],
    [(200, 100, p) for p in (0.01:0.01:0.05)],
)

symmetric_params = vcat(
    [(10, p) for p in (0.0:0.1:0.5)], #
    [(100, p) for p in (0.01:0.01:0.05)],
)

@testset "Column coloring & decompression" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)
    algo = GreedyColoringAlgorithm(; decompression=:direct)
    @testset "$((; m, n, p))" for (m, n, p) in asymmetric_params
        A0 = sprand(rng, m, n, p)
        color0 = column_coloring(A0, algo)
        @test structurally_orthogonal_columns(A0, color0)
        @test directly_recoverable_columns(A0, color0)
        test_coloring_decompression(A0, problem, algo; color0)
    end
    @testset "$((; n, p))" for (n, p) in symmetric_params
        A0 = sparse(Symmetric(sprand(rng, n, n, p)))
        color0 = column_coloring(A0, algo)
        @test structurally_orthogonal_columns(A0, color0)
        @test directly_recoverable_columns(A0, color0)
        test_coloring_decompression(A0, problem, algo; color0)
    end
end;

@testset "Row coloring & decompression" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:row)
    algo = GreedyColoringAlgorithm(; decompression=:direct)
    @testset "$((; m, n, p))" for (m, n, p) in asymmetric_params
        A0 = sprand(rng, m, n, p)
        color0 = row_coloring(A0, algo)
        @test structurally_orthogonal_columns(transpose(A0), color0)
        @test directly_recoverable_columns(transpose(A0), color0)
        test_coloring_decompression(A0, problem, algo; color0)
    end
    @testset "$((; n, p))" for (n, p) in symmetric_params
        A0 = sparse(Symmetric(sprand(rng, n, n, p)))
        color0 = row_coloring(A0, algo)
        @test structurally_orthogonal_columns(transpose(A0), color0)
        @test directly_recoverable_columns(transpose(A0), color0)
        test_coloring_decompression(A0, problem, algo; color0)
    end
end;

@testset "Symmetric coloring & direct decompression" begin
    problem = ColoringProblem(; structure=:symmetric, partition=:column)
    algo = GreedyColoringAlgorithm(; decompression=:direct)
    @testset "$((; n, p))" for (n, p) in symmetric_params
        A0 = sparse(Symmetric(sprand(rng, n, n, p)))
        color0 = symmetric_coloring(A0, algo)
        @test symmetrically_orthogonal_columns(A0, color0)
        @test directly_recoverable_columns(A0, color0)
        test_coloring_decompression(A0, problem, algo; color0)
    end
end;

@testset "Symmetric coloring & substitution decompression" begin
    problem = ColoringProblem(; structure=:symmetric, partition=:column)
    algo = GreedyColoringAlgorithm(; decompression=:substitution)
    @testset "$((; n, p))" for (n, p) in symmetric_params
        A0 = sparse(Symmetric(sprand(rng, n, n, p)))
        # TODO: find tests for recoverability
        test_coloring_decompression(A0, problem, algo)
    end
end;
