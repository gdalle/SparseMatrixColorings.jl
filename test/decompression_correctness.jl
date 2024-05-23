using ADTypes: column_coloring, row_coloring, symmetric_coloring
using Compat
using SparseArrays
using SparseMatrixColorings
using SparseMatrixColorings: color_groups, decompress_columns, decompress_rows
using StableRNGs
using Test

rng = StableRNG(63)

algo = GreedyColoringAlgorithm()

m, n = 10, 20

A0 = sprand(rng, Bool, m, n, 0.3)
A1 = Matrix(A0)
A0t = transpose(A0)
A1t = transpose(A1)

S0 = map(!iszero, A0)
S1 = map(!iszero, A1)
S0t = transpose(S0)
S1t = transpose(S1)

@testset "Column decompression" begin
    @testset "$(typeof(A))" for (A, S) in zip((A0, A1), (S0, S1))
        colors = column_coloring(A, algo)
        groups = color_groups(colors)
        @test length(groups[1]) > 1
        C = stack(groups) do group
            dropdims(sum(A[:, group]; dims=2); dims=2)
        end
        A_new = decompress_columns(S, C, colors)
        @test A_new == A
    end
end

@testset "Row decompression" begin
    @testset "$(typeof(At))" for (At, St) in zip((A0t, A1t), (S0t, S1t))
        colors = row_coloring(At, algo)
        groups = color_groups(colors)
        @test length(groups[1]) > 1
        Ct = stack(groups; dims=1) do group
            dropdims(sum(At[group, :]; dims=1); dims=1)
        end
        At_new = decompress_rows(St, Ct, colors)
        @test At_new == At
    end
end
