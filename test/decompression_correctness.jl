using Base.Iterators: product
using ADTypes: column_coloring, row_coloring, symmetric_coloring
using Compat
using LinearAlgebra
using SparseArrays
using SparseMatrixColorings
using SparseMatrixColorings:
    color_groups,
    decompress_columns,
    decompress_columns!,
    decompress_rows,
    decompress_rows!,
    decompress_symmetric,
    decompress_symmetric!,
    matrix_versions,
    same_sparsity_pattern
using StableRNGs
using Test

rng = StableRNG(63)

algo = GreedyColoringAlgorithm()

@testset "Column decompression" begin
    @testset "Small" begin
        A0 = [
            1 0 2
            0 3 4
            5 0 0
        ]
        S0 = Bool[
            1 0 1
            0 1 1
            1 0 0
        ]
        C = [
            1 2
            3 4
            5 0
        ]
        colors = [1, 1, 2]
        @testset "A::$(typeof(A)) - S::$(typeof(S))" for (A, S) in product(
            matrix_versions(A0), matrix_versions(S0)
        )
            @test decompress_columns(S, C, colors) == A
        end
    end

    @testset "Medium" begin
        m, n = 18, 20
        A0 = sprand(rng, Bool, m, n, 0.2)
        S0 = map(!iszero, A0)
        colors = column_coloring(A0, algo)
        groups = color_groups(colors)
        C = stack(groups; dims=2) do group
            dropdims(sum(A0[:, group]; dims=2); dims=2)
        end
        @test size(C) == (size(A0, 1), length(groups))
        @testset "A::$(typeof(A)) - S::$(typeof(S))" for (A, S) in product(
            matrix_versions(A0), matrix_versions(S0)
        )
            @test decompress_columns(S, C, colors) == A
        end
    end
end;

@testset "Row decompression" begin
    @testset "Small" begin
        A0 = [
            1 0 3
            0 2 0
            4 5 0
        ]
        S0 = Bool[
            1 0 1
            0 1 0
            1 1 0
        ]
        C = [
            1 2 3
            4 5 0
        ]
        colors = [1, 1, 2]
        @testset "A::$(typeof(A)) - S::$(typeof(S))" for (A, S) in product(
            matrix_versions(A0), matrix_versions(S0)
        )
            @test decompress_rows(S, C, colors) == A
        end
    end

    @testset "Medium" begin
        m, n = 18, 20
        A0 = sprand(rng, Bool, m, n, 0.2)
        S0 = map(!iszero, A0)
        colors = row_coloring(A0, algo)
        groups = color_groups(colors)
        C = stack(groups; dims=1) do group
            dropdims(sum(A0[group, :]; dims=1); dims=1)
        end
        @test size(C) == (length(groups), size(A0, 2))
        @testset "A::$(typeof(A)) - S::$(typeof(S))" for (A, S) in product(
            matrix_versions(A0), matrix_versions(S0)
        )
            @test decompress_rows(S, C, colors) == A
        end
    end
end;

@testset "Symmetric decompression" begin
    @testset "Small" begin
        # Fig 4.1 from "What color is your Jacobian?"
        A0 = [
            1 2 0 0 0 0
            2 3 4 0 5 6
            0 4 7 8 0 0
            0 0 8 9 0 10
            0 5 0 0 11 0
            0 6 0 10 0 12
        ]
        S0 = (!iszero).(A0)
        colors = [
            1,  # green
            2,  # red
            1,  # green
            3,  # blue
            1,  # green
            1,  # green
        ]
        groups = color_groups(colors)
        C = stack(groups; dims=2) do group
            dropdims(sum(A0[:, group]; dims=2); dims=2)
        end
        @testset "A::$(typeof(A)) - S::$(typeof(S))" for (A, S) in product(
            matrix_versions(A0), matrix_versions(S0)
        )
            @test decompress_symmetric(S, C, colors) == A
        end
    end
end;
