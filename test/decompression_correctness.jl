using ADTypes: column_coloring, row_coloring, symmetric_coloring
using Compat
using SparseArrays
using SparseMatrixColorings: color_groups, decompress_columns, decompress_rows
using StableRNGs
using Test

rng = StableRNG(63)

algo = GreedyColoringAlgorithm()

m, n = 10, 20
A = sprand(rng, Bool, m, n, 0.3)
At = transpose(A)
S = map(!iszero, A)
St = transpose(S)

@testset "Column decompression" begin
    colors = column_coloring(A, algo)
    groups = color_groups(colors)
    @test length(groups[1]) > 1
    C = stack(groups) do group
        dropdims(sum(A[:, group]; dims=2); dims=2)
    end
    A_new = decompress_columns(S, C, colors)
    @test A_new == A
end

@testset "Row decompression" begin
    colors = row_coloring(At, algo)
    groups = color_groups(colors)
    @test length(groups[1]) > 1
    Ct = stack(groups; dims=1) do group
        dropdims(sum(At[group, :]; dims=1); dims=1)
    end
    At_new = decompress_rows(St, Ct, colors)
    @test At_new == At
end
