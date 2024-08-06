using Base.Iterators: product
using Compat
using LinearAlgebra: I, Symmetric
using SparseArrays: sprand
using SparseMatrixColorings
using SparseMatrixColorings:
    GreedyColoringAlgorithm,
    structurally_orthogonal_columns,
    symmetrically_orthogonal_columns,
    directly_recoverable_columns,
    matrix_versions
using StableRNGs
using Test

rng = StableRNG(63)

algo = GreedyColoringAlgorithm()

@test startswith(string(algo), "GreedyColoringAlgorithm(")

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
    @testset "Size ($m, $n) - sparsity $p" for (m, n, p) in asymmetric_params
        A0 = sprand(rng, Bool, m, n, p)
        S0 = map(!iszero, A0)
        @testset "A::$(typeof(A))" for A in matrix_versions(A0)
            coloring_result = column_coloring_detailed(A, algo)
            color = get_colors(coloring_result)
            @test structurally_orthogonal_columns(A, color)
            @test directly_recoverable_columns(A, color)
            group = color_groups(color)
            B = stack(group; dims=2) do g
                dropdims(sum(A[:, g]; dims=2); dims=2)
            end
            @testset "S::$(typeof(S))" for S in matrix_versions(S0)
                @test decompress_columns(S, B, coloring_result) == A
            end
        end
    end
end;

@testset "Row coloring & decompression" begin
    @testset "Size ($m, $n) - sparsity $p" for (m, n, p) in asymmetric_params
        A0 = sprand(rng, Bool, m, n, p)
        S0 = map(!iszero, A0)
        @testset "A::$(typeof(A))" for A in matrix_versions(A0)
            coloring_result = row_coloring_detailed(A, algo)
            color = get_colors(coloring_result)
            @test structurally_orthogonal_columns(transpose(A), color)
            @test directly_recoverable_columns(transpose(A), color)
            group = color_groups(color)
            B = stack(group; dims=1) do g
                dropdims(sum(A[g, :]; dims=1); dims=1)
            end
            @testset "S::$(typeof(S))" for S in matrix_versions(S0)
                @test decompress_rows(S, B, coloring_result) == A
            end
        end
    end
end;

@testset "Symmetric coloring & decompression" begin
    @testset "Size ($n, $n) - sparsity $p" for (n, p) in symmetric_params
        A0 = Symmetric(sprand(rng, Bool, n, n, p))
        S0 = map(!iszero, A0)
        @testset "A::$(typeof(A))" for A in matrix_versions(A0)
            # Naive decompression
            coloring_result = symmetric_coloring_detailed(A, algo)
            color = get_colors(coloring_result)
            @test color == symmetric_coloring(A, algo)
            @test symmetrically_orthogonal_columns(A, color)
            @test directly_recoverable_columns(A, color)
            group = color_groups(color)
            B = stack(group; dims=2) do g
                dropdims(sum(A[:, g]; dims=2); dims=2)
            end
            @testset "S::$(typeof(S))" for S in matrix_versions(S0)
                @test decompress_symmetric(S, B, coloring_result) == A
            end
        end
    end
end;
