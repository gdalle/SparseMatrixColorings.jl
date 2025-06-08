using CUDA.CUSPARSE: CuSparseMatrixCSC, CuSparseMatrixCSR
using LinearAlgebra
using SparseArrays
using SparseMatrixColorings
using StableRNGs
using Test

rng = StableRNG(63)

asymmetric_params = vcat(
    [(10, 20, p) for p in (0.0:0.2:0.5)],
    [(20, 10, p) for p in (0.0:0.2:0.5)],
    [(100, 200, p) for p in (0.01:0.02:0.05)],
    [(200, 100, p) for p in (0.01:0.02:0.05)],
)

symmetric_params = vcat(
    [(10, p) for p in (0.0:0.2:0.5)], #
    [(100, p) for p in (0.01:0.02:0.05)],
)

@testset "Column coloring & decompression" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)
    algo = GreedyColoringAlgorithm(; decompression=:direct)
    @testset for T in (CuSparseMatrixCSC,)
        @testset "$((; m, n, p))" for (m, n, p) in asymmetric_params
            A0 = T(sprand(rng, m, n, p))
            test_coloring_decompression(A0, problem, algo; gpu=true)
        end
    end
end;

@testset "Row coloring & decompression" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:row)
    algo = GreedyColoringAlgorithm(; decompression=:direct)
    @testset for T in (CuSparseMatrixCSC,)
        @testset "$((; m, n, p))" for (m, n, p) in asymmetric_params
            A0 = T(sprand(rng, m, n, p))
            test_coloring_decompression(A0, problem, algo; gpu=true)
        end
    end
end;

@testset "Symmetric coloring & direct decompression" begin
    problem = ColoringProblem(; structure=:symmetric, partition=:column)
    algo = GreedyColoringAlgorithm(; postprocessing=false, decompression=:direct)
    @testset for T in (CuSparseMatrixCSC,)
        @testset "$((; n, p))" for (n, p) in symmetric_params
            A0 = T(sparse(Symmetric(sprand(rng, n, n, p))))
            test_coloring_decompression(A0, problem, algo; gpu=true)
        end
    end
end;
