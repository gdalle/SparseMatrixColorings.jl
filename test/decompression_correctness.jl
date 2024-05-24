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
    same_sparsity_pattern
using StableRNGs
using Test

rng = StableRNG(63)

algo = GreedyColoringAlgorithm()

@testset "Sparsity pattern comparison" begin
    A = [
        1 1
        0 1
        0 0
    ]
    B1 = [
        1 1
        0 1
        1 0
    ]
    B2 = [
        1 1
        0 0
        0 1
    ]
    C = [
        1 1 0
        0 1 0
        0 0 0
    ]

    @test same_sparsity_pattern(sparse(A), sparse(A))
    @test !same_sparsity_pattern(sparse(A), sparse(B1))
    @test_broken !same_sparsity_pattern(sparse(A), sparse(B2))
    @test !same_sparsity_pattern(sparse(A), sparse(C))

    @test same_sparsity_pattern(transpose(sparse(A)), transpose(sparse(A)))
    @test !same_sparsity_pattern(transpose(sparse(A)), transpose(sparse(B1)))
    @test_broken !same_sparsity_pattern(transpose(sparse(A)), transpose(sparse(B2)))
    @test !same_sparsity_pattern(transpose(sparse(A)), transpose(sparse(C)))
end;

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
        @testset "$(typeof(A)) - $(typeof(S))" for (A, S) in [
            (A0, S0),  #
            (sparse(A0), sparse(S0)),
        ]
            @test decompress_columns(S, C, colors) == A
            if A isa SparseMatrixCSC
                @test_throws DimensionMismatch decompress_columns!(
                    similar(A), false .* S, C, colors
                )
            end
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
        @testset "$(typeof(A)) - $(typeof(S))" for (A, S) in [
            (A0, S0),  #
            (Matrix(A0), Matrix(S0)),
        ]
            @test decompress_columns(S, C, colors) == A
        end
    end
end

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
        @testset "$(typeof(A)) - $(typeof(S))" for (A, S) in [
            (A0, S0), #
            (transpose(sparse(transpose(A0))), transpose(sparse(transpose(S0)))),
        ]
            @test decompress_rows(S, C, colors) == A
            if A isa Transpose{<:Any,<:SparseMatrixCSC}
                @test_throws DimensionMismatch decompress_rows!(
                    transpose(similar(parent(A))), transpose(false .* parent(S)), C, colors
                )
            end
        end
    end
    @testset "Medium" begin
        m, n = 18, 20
        A0 = sprand(rng, Bool, m, n, 0.2)
        S0 = map(!iszero, A0)
        A0t = transpose(A0)
        S0t = transpose(S0)
        colors = row_coloring(A0t, algo)
        groups = color_groups(colors)
        Ct = stack(groups; dims=1) do group
            dropdims(sum(A0t[group, :]; dims=1); dims=1)
        end
        @test size(Ct) == (length(groups), size(A0t, 2))
        @testset "$(typeof(At)) - $(typeof(St))" for (At, St) in [
            (A0t, S0t),  #
            (Matrix(A0t), Matrix(S0t)),
        ]
            @test decompress_rows(St, Ct, colors) == At
        end
    end
end
