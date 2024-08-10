using ADTypes: column_coloring, row_coloring, symmetric_coloring
using Base.Iterators: product
using Compat
using LinearAlgebra: I, Symmetric
using SparseArrays
using SparseMatrixColorings
using SparseMatrixColorings:
    DefaultColoringResult,
    structurally_orthogonal_columns,
    symmetrically_orthogonal_columns,
    directly_recoverable_columns,
    matrix_versions,
    respectful_similar
using StableRNGs
using Test

rng = StableRNG(63)

algo = GreedyColoringAlgorithm()

asymmetric_params = vcat(
    [(10, 20, p) for p in (0.1:0.1:0.5)],
    [(20, 10, p) for p in (0.1:0.1:0.5)],
    [(100, 200, p) for p in (0.01:0.01:0.05)],
    [(200, 100, p) for p in (0.01:0.01:0.05)],
)

symmetric_params = vcat(
    [(10, p) for p in (0.1:0.1:0.5)], #
    [(100, p) for p in (0.01:0.01:0.05)],
)

@testset "Column coloring & decompression" begin
    problem = ColoringProblem(;
        structure=:nonsymmetric, partition=:column, decompression=:direct
    )
    @testset "Size ($m, $n) - sparsity $p" for (m, n, p) in asymmetric_params
        A0 = sprand(rng, m, n, p)
        color0 = column_coloring(A0, algo)
        @test structurally_orthogonal_columns(A0, color0)
        @test directly_recoverable_columns(A0, color0)
        test_coloring_decompression(A0, problem, algo; color0)
    end
end;

@testset "Row coloring & decompression" begin
    problem = ColoringProblem(;
        structure=:nonsymmetric, partition=:row, decompression=:direct
    )
    @testset "Size ($m, $n) - sparsity $p" for (m, n, p) in asymmetric_params
        A0 = sprand(rng, m, n, p)
        color0 = row_coloring(A0, algo)
        @test structurally_orthogonal_columns(transpose(A0), color0)
        @test directly_recoverable_columns(transpose(A0), color0)
        test_coloring_decompression(A0, problem, algo; color0)
    end
end;

@testset "Symmetric coloring & direct decompression" begin
    problem = ColoringProblem(;
        structure=:symmetric, partition=:column, decompression=:direct
    )
    @testset "Size ($n, $n) - sparsity $p" for (n, p) in symmetric_params
        A0 = Symmetric(sprand(rng, n, n, p))
        color0 = symmetric_coloring(A0, algo)
        @test symmetrically_orthogonal_columns(A0, color0)
        @test directly_recoverable_columns(A0, color0)
        test_coloring_decompression(A0, problem, algo; color0)
    end
end;

@testset "Symmetric coloring & substitution decompression" begin
    problem = ColoringProblem(;
        structure=:symmetric, partition=:column, decompression=:substitution
    )
    @testset "Size ($n, $n) - sparsity $p" for (n, p) in symmetric_params
        A0 = Symmetric(sprand(rng, n, n, p))
        color0 = column_colors(coloring(A0, problem, algo))
        # TODO: find tests for recoverability
        test_coloring_decompression(A0, problem, algo; color0)
    end
end;
