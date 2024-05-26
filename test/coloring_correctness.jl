using LinearAlgebra: I, Symmetric
using SparseArrays: sprand
using SparseMatrixColorings
using SparseMatrixColorings:
    GreedyColoringAlgorithm,
    check_structurally_orthogonal_columns,
    check_symmetrically_orthogonal_columns,
    matrix_versions
using StableRNGs
using Test

rng = StableRNG(63)

algo = GreedyColoringAlgorithm()

@test startswith(string(algo), "GreedyColoringAlgorithm(")

asymmetric_params = vcat(
    [(10, 20, p) for p in (0.1:0.1:0.5)],
    [(20, 10, p) for p in (0.1:0.1:0.5)],
    [(100, 200, p) for p in (0.01:0.01:0.05)],
    [(200, 100, p) for p in (0.01:0.01:0.05)],
)

symmetric_params = vcat(
    [(10, p) for p in (0.1:0.05:0.5)], #
    [(100, p) for p in (0.01:0.005:0.05)],
)

@testset "Column coloring" begin
    @testset "Size ($m, $n) - sparsity $p" for (m, n, p) in asymmetric_params
        @testset "$(typeof(A))" for A in matrix_versions(sprand(rng, Bool, m, n, p))
            column_color = column_coloring(A, algo)
            @test check_structurally_orthogonal_columns(A, column_color)
        end
    end
end;

@testset "Row coloring" begin
    @testset "Size ($m, $n) - sparsity $p" for (m, n, p) in asymmetric_params
        @testset "$(typeof(A))" for A in matrix_versions(sprand(rng, Bool, m, n, p))
            row_color = row_coloring(A, algo)
            @test check_structurally_orthogonal_columns(transpose(A), row_color)
        end
    end
end;

@testset "Symmetric coloring" begin
    @testset "Size ($n, $n) - sparsity $p" for (n, p) in symmetric_params
        @testset "$(typeof(A))" for A in
                                    matrix_versions(Symmetric(sprand(rng, Bool, n, n, p)))
            symmetric_color = symmetric_coloring(A, algo)
            @test check_symmetrically_orthogonal_columns(A, symmetric_color)
        end
    end
end;
