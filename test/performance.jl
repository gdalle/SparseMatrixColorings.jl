using ADTypes: column_coloring, row_coloring, symmetric_coloring
using Chairmarks
using JET
using LinearAlgebra
using SparseArrays
using SparseMatrixColorings
using SparseMatrixColorings:
    adjacency_graph,
    bipartite_graph,
    decompress!,
    partial_distance2_coloring!,
    respectful_similar
using StableRNGs
using Test

rng = StableRNG(63)

@testset "Coloring - type stability" begin
    n = 10
    A = sprand(rng, Bool, n, n, 3 / n)
    algo = GreedyColoringAlgorithm()
    @test_opt target_modules = (SparseMatrixColorings,) column_coloring(A, algo)
    @test_opt target_modules = (SparseMatrixColorings,) row_coloring(A, algo)
    @test_opt target_modules = (SparseMatrixColorings,) symmetric_coloring(
        Symmetric(A), algo
    )
    @test_opt target_modules = (SparseMatrixColorings,) coloring(A, ColoringProblem(), algo)
end

function benchmark_distance2_coloring(n)
    return @be (;
        bg=bipartite_graph(sprand(Bool, n, n, 3 / n)),
        color=Vector{Int}(undef, n),
        forbidden_colors=Vector{Int}(undef, n),
    ) partial_distance2_coloring!(_.color, _.forbidden_colors, _.bg, Val(1), 1:n) evals = 1
end

@testset "Coloring - allocations" begin
    @test minimum(benchmark_distance2_coloring(10)).allocs == 0
    @test minimum(benchmark_distance2_coloring(100)).allocs == 0
    @test minimum(benchmark_distance2_coloring(1000)).allocs == 0
end

function benchmark_sparse_decompression(n)
    A = sprand(n, 2n, 5 / n)
    result = coloring(
        A,
        ColoringProblem(; structure=:nonsymmetric, partition=:column),
        GreedyColoringAlgorithm(),
    )
    group = column_groups(result)
    B = stack(group; dims=2) do g
        dropdims(sum(A[:, g]; dims=2); dims=2)
    end
    return @be respectful_similar(A) decompress!(_, B, result) evals = 1
end

@testset "SparseMatrixCSC decompression - allocations" begin
    @test minimum(benchmark_sparse_decompression(10)).allocs == 0
    @test minimum(benchmark_sparse_decompression(100)).allocs == 0
    @test minimum(benchmark_sparse_decompression(1000)).allocs == 0
end
