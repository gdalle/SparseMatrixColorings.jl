using SparseArrays
using SparseMatrixColorings
using StableRNGs
using Test

rng = StableRNG(0)

asymmetric_params = vcat(
    [(10, 20, p) for p in (0.0:0.1:0.5)],
    [(20, 10, p) for p in (0.0:0.1:0.5)],
    [(100, 200, p) for p in (0.01:0.01:0.05)],
    [(200, 100, p) for p in (0.01:0.01:0.05)],
)

@testset "Column coloring" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)
    algo = GreedyColoringAlgorithm()
    optalgo = OptimalColoringAlgorithm(HiGHS.Optimizer)
    for (m, n, p) in asymmetric_params
        A = sprand(rng, m, n, p)
        result = coloring(A, problem, algo)
        optresult = coloring(A, problem, optalgo)
        @test ncolors(result) >= ncolors(optresult)
    end
end

@testset "Row coloring" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:row)
    algo = GreedyColoringAlgorithm()
    optalgo = OptimalColoringAlgorithm(HiGHS.Optimizer)
    for (m, n, p) in asymmetric_params
        A = sprand(rng, m, n, p)
        result = coloring(A, problem, algo)
        optresult = coloring(A, problem, optalgo)
        @test ncolors(result) >= ncolors(optresult)
    end
end
