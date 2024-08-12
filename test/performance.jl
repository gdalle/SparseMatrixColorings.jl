using ADTypes: column_coloring, row_coloring, symmetric_coloring
using Chairmarks
using JET
using LinearAlgebra
using SparseArrays
using SparseMatrixColorings
using SparseMatrixColorings:
    adjacency_graph, bipartite_graph, partial_distance2_coloring!, respectful_similar
using StableRNGs
using Test

rng = StableRNG(63)

@testset "Coloring - type stability" begin
    n = 10
    A = sprand(rng, Bool, n, n, 3 / n)

    # ADTypes
    @test_opt target_modules = (SparseMatrixColorings,) column_coloring(
        A, GreedyColoringAlgorithm()
    )
    @test_opt target_modules = (SparseMatrixColorings,) row_coloring(
        A, GreedyColoringAlgorithm()
    )
    @test_opt target_modules = (SparseMatrixColorings,) symmetric_coloring(
        Symmetric(A), GreedyColoringAlgorithm()
    )

    # Native
    @test_opt target_modules = (SparseMatrixColorings,) coloring(
        A,
        ColoringProblem(; structure=:nonsymmetric, partition=:column),
        GreedyColoringAlgorithm(; decompression=:direct),
    )
    @test_opt target_modules = (SparseMatrixColorings,) coloring(
        A,
        ColoringProblem(; structure=:nonsymmetric, partition=:row),
        GreedyColoringAlgorithm(; decompression=:direct),
    )
    @test_opt target_modules = (SparseMatrixColorings,) coloring(
        A,
        ColoringProblem(; structure=:symmetric, partition=:column),
        GreedyColoringAlgorithm(; decompression=:direct),
    )
    @test_opt target_modules = (SparseMatrixColorings,) coloring(
        A,
        ColoringProblem(; structure=:symmetric, partition=:column),
        GreedyColoringAlgorithm(; decompression=:substitution),
    )
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

function benchmark_sparse_decompression(
    n::Integer; structure::Symbol, partition::Symbol, decompression::Symbol
)
    A = if structure == :nonsymmetric
        sprand(n, 2n, 5 / n)
    elseif structure == :symmetric
        sparse(Symmetric(sprand(n, n, 5 / n)))
    end
    result = coloring(
        A, ColoringProblem(; structure, partition), GreedyColoringAlgorithm(; decompression)
    )
    B = compress(A, result)
    return @be respectful_similar(A) decompress!(_, B, result) evals = 1
end

@testset "SparseMatrixCSC decompression - allocations" begin
    @test minimum(
        benchmark_sparse_decompression(
            1000; structure=:nonsymmetric, partition=:column, decompression=:direct
        ),
    ).allocs == 0
    @test minimum(
        benchmark_sparse_decompression(
            1000; structure=:nonsymmetric, partition=:row, decompression=:direct
        ),
    ).allocs == 0
    @test minimum(
        benchmark_sparse_decompression(
            1000; structure=:symmetric, partition=:column, decompression=:direct
        ),
    ).allocs == 0
    @test minimum(
        benchmark_sparse_decompression(
            1000; structure=:symmetric, partition=:column, decompression=:substitution
        ),
    ).allocs == 0
end
