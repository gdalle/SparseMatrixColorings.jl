using Chairmarks
using JET
using LinearAlgebra
using SparseArrays
using SparseMatrixColorings
using SparseMatrixColorings: adjacency_graph, bipartite_graph, partial_distance2_coloring!
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
end

function benchmark_distance2_coloring(n)
    @be (;
        bg=bipartite_graph(sprand(Bool, n, n, 3 / n)),
        color=Vector{Int}(undef, n),
        forbidden_colors=Vector{Int}(undef, n),
    ) partial_distance2_coloring!(_.color, _.forbidden_colors, _.bg, Val(1), 1:n) evals = 1
end

@testset "Coloring - allocations" begin
    @test minimum(benchmark_distance2_coloring(10)).allocs == 0
end
