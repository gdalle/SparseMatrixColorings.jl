using ADTypes: column_coloring, row_coloring, symmetric_coloring
using Chairmarks
using JET
using LinearAlgebra
using SparseArrays
using SparseMatrixColorings:
    adjacency_graph, bipartite_graph, partial_distance2_coloring!, star_coloring1!
using StableRNGs
using Test

rng = StableRNG(63)

@testset "Type stability" begin
    n = 10
    A = sprand(rng, Bool, n, n, 3 / n)
    algo = GreedyColoringAlgorithm()
    @test_opt target_modules = (SparseMatrixColorings,) column_coloring(A, algo)
    @test_opt target_modules = (SparseMatrixColorings,) row_coloring(A, algo)
    @test_opt target_modules = (SparseMatrixColorings,) symmetric_coloring(
        A + transpose(A), algo
    )
end

function benchmark_distance2_coloring(n)
    @be (;
        bg=bipartite_graph(sprand(Bool, n, n, 3 / n)),
        colors=Vector{Int}(undef, n),
        forbidden_colors=Vector{Int}(undef, n),
    ) partial_distance2_coloring!(_.colors, _.forbidden_colors, _.bg, Val(1), 1:n) evals = 1
end

function benchmark_star_coloring(n)
    @be (;
        g=adjacency_graph(Symmetric(sprand(Bool, n, n, 3 / n))),
        colors=Vector{Int}(undef, n),
        forbidden_colors=Vector{Int}(undef, n),
    ) star_coloring1!(_.colors, _.forbidden_colors, _.g, 1:n) evals = 1
end

@testset "Allocations" begin
    @test minimum(benchmark_distance2_coloring(10)).allocs == 0
    @test minimum(benchmark_star_coloring(10)).allocs == 0
end
