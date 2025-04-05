using ADTypes: column_coloring, row_coloring, symmetric_coloring
using LinearAlgebra
using SparseArrays
using SparseMatrixColorings
using StableRNGs
using Test

rng = StableRNG(63)

asymmetric_params = vcat(
    [(10, 20, p) for p in (0.0:0.1:0.5)],
    [(20, 10, p) for p in (0.0:0.1:0.5)],
    [(100, 200, p) for p in (0.01:0.01:0.05)],
    [(200, 100, p) for p in (0.01:0.01:0.05)],
)

symmetric_params = vcat(
    [(10, p) for p in (0.0:0.1:0.5)], #
    [(100, p) for p in (0.01:0.01:0.05)],
)

@testset "Column coloring & decompression" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)
    algo = GreedyColoringAlgorithm(; decompression=:direct, allow_denser=true)
    @testset "$((; m, n, p))" for (m, n, p) in asymmetric_params
        A0 = sprand(rng, m, n, p)
        color0 = column_coloring(A0, algo)
        test_coloring_decompression(A0, problem, algo; color0)
    end
    @testset "$((; n, p))" for (n, p) in symmetric_params
        A0 = sparse(Symmetric(sprand(rng, n, n, p)))
        color0 = column_coloring(A0, algo)
        test_coloring_decompression(A0, problem, algo; color0)
    end
end;

@testset "Row coloring & decompression" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:row)
    algo = GreedyColoringAlgorithm(; decompression=:direct, allow_denser=true)
    @testset "$((; m, n, p))" for (m, n, p) in asymmetric_params
        A0 = sprand(rng, m, n, p)
        color0 = row_coloring(A0, algo)
        test_coloring_decompression(A0, problem, algo; color0)
    end
    @testset "$((; n, p))" for (n, p) in symmetric_params
        A0 = sparse(Symmetric(sprand(rng, n, n, p)))
        color0 = row_coloring(A0, algo)
        test_coloring_decompression(A0, problem, algo; color0)
    end
end;

@testset "Symmetric coloring & direct decompression" begin
    problem = ColoringProblem(; structure=:symmetric, partition=:column)
    @testset for algo in (
        GreedyColoringAlgorithm(;
            postprocessing=false, decompression=:direct, allow_denser=true
        ),
        GreedyColoringAlgorithm(;
            postprocessing=true, decompression=:direct, allow_denser=true
        ),
    )
        @testset "$((; n, p))" for (n, p) in symmetric_params
            A0 = sparse(Symmetric(sprand(rng, n, n, p)))
            color0 = algo.postprocessing ? nothing : symmetric_coloring(A0, algo)
            test_coloring_decompression(A0, problem, algo; color0)
        end
    end
end;

@testset "Symmetric coloring & substitution decompression" begin
    problem = ColoringProblem(; structure=:symmetric, partition=:column)
    @testset for algo in (
        GreedyColoringAlgorithm(; postprocessing=false, decompression=:substitution),
        GreedyColoringAlgorithm(; postprocessing=true, decompression=:substitution),
    )
        @testset "$((; n, p))" for (n, p) in symmetric_params
            A0 = sparse(Symmetric(sprand(rng, n, n, p)))
            # TODO: find tests for recoverability
            test_coloring_decompression(A0, problem, algo)
        end
    end
end;

@testset "Bicoloring & direct decompression" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:bidirectional)
    @testset for algo in (
        GreedyColoringAlgorithm(
            RandomOrder(rng); postprocessing=false, decompression=:direct
        ),
        GreedyColoringAlgorithm(
            RandomOrder(rng); postprocessing=true, decompression=:direct
        ),
    )
        @testset "$((; m, n, p))" for (m, n, p) in asymmetric_params
            A0 = sprand(rng, m, n, p)
            test_bicoloring_decompression(A0, problem, algo)
        end
        @testset "$((; n, p))" for (n, p) in symmetric_params
            A0 = sparse(Symmetric(sprand(rng, n, n, p)))
            test_bicoloring_decompression(A0, problem, algo)
        end
    end
end;

@testset "Bicoloring & substitution decompression" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:bidirectional)
    @testset for algo in (
        GreedyColoringAlgorithm(
            RandomOrder(rng); postprocessing=false, decompression=:substitution
        ),
        GreedyColoringAlgorithm(
            RandomOrder(rng); postprocessing=true, decompression=:substitution
        ),
    )
        @testset "$((; m, n, p))" for (m, n, p) in asymmetric_params
            A0 = sprand(rng, m, n, p)
            test_bicoloring_decompression(A0, problem, algo)
        end
        @testset "$((; n, p))" for (n, p) in symmetric_params
            A0 = sparse(Symmetric(sprand(rng, n, n, p)))
            test_bicoloring_decompression(A0, problem, algo)
        end
    end
end;
