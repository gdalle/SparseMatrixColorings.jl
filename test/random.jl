using Base.Iterators: product
using Compat
using LinearAlgebra: I, Symmetric
using SparseArrays: sprand
using SparseMatrixColorings
using SparseMatrixColorings:
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
        @testset "A::$(typeof(A))" for A in matrix_versions(A0)
            result = coloring(A, problem, algo)
            color = column_colors(result)
            group = column_groups(result)
            @test color == column_coloring(A, algo)
            @test structurally_orthogonal_columns(A, color)
            @test directly_recoverable_columns(A, color)
            B = stack(group; dims=2) do g
                dropdims(sum(A[:, g]; dims=2); dims=2)
            end
            @test decompress(B, result) == A
            @test decompress!(respectful_similar(A), B, result) == A
        end
    end
end;

@testset "Row coloring & decompression" begin
    problem = ColoringProblem(;
        structure=:nonsymmetric, partition=:row, decompression=:direct
    )
    @testset "Size ($m, $n) - sparsity $p" for (m, n, p) in asymmetric_params
        A0 = sprand(rng, m, n, p)
        @testset "A::$(typeof(A))" for A in matrix_versions(A0)
            result = coloring(A, problem, algo)
            color = row_colors(result)
            group = row_groups(result)
            @test color == row_coloring(A, algo)
            @test structurally_orthogonal_columns(transpose(A), color)
            @test directly_recoverable_columns(transpose(A), color)
            B = stack(group; dims=1) do g
                dropdims(sum(A[g, :]; dims=1); dims=1)
            end
            @test decompress(B, result) == A
            @test decompress!(respectful_similar(A), B, result) == A
        end
    end
end;

@testset "Symmetric coloring & decompression" begin
    problem = ColoringProblem(;
        structure=:symmetric, partition=:column, decompression=:direct
    )
    @testset "Size ($n, $n) - sparsity $p" for (n, p) in symmetric_params
        A0 = Symmetric(sprand(rng, n, n, p))
        @testset "A::$(typeof(A))" for A in matrix_versions(A0)
            # Naive decompression
            result = coloring(A, problem, algo)
            color = column_colors(result)
            group = column_groups(result)
            @test color == symmetric_coloring(A, algo)
            @test symmetrically_orthogonal_columns(A, color)
            @test directly_recoverable_columns(A, color)
            B = stack(group; dims=2) do g
                dropdims(sum(A[:, g]; dims=2); dims=2)
            end
            @test decompress(B, result) == A
            @test decompress!(respectful_similar(A), B, result) == A
        end
    end
end;
