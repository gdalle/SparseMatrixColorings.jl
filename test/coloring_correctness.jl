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

@testset "Column coloring" begin
    @testset "Size ($m, $n) - sparsity $p" for (m, n, p) in [
        (10, 20, 0.05), (20, 10, 0.05), (100, 200, 0.05), (200, 100, 0.05)
    ]
        @testset "$(typeof(A))" for A in matrix_versions(sprand(rng, Bool, m, n, p))
            column_color = column_coloring(A, algo)
            @test check_structurally_orthogonal_columns(A, column_color)
            @test minimum(column_color) == 1
            @test maximum(column_color) < size(A, 2) ÷ 2
        end
    end
end;

@testset "Row coloring" begin
    @testset "Size ($m, $n) - sparsity $p" for (m, n, p) in [
        (10, 20, 0.05), (20, 10, 0.05), (100, 200, 0.05), (200, 100, 0.05)
    ]
        @testset "$(typeof(A))" for A in matrix_versions(sprand(rng, Bool, m, n, 0.05))
            row_color = row_coloring(A, algo)
            @test check_structurally_orthogonal_columns(transpose(A), row_color)
            @test minimum(row_color) == 1
            @test maximum(row_color) < size(A, 1) ÷ 2
        end
    end
end;

@testset "Symmetric coloring" begin
    @testset "Size ($n, $n) - sparsity $p" for (n, p) in [(10, 0.05), (100, 0.05)]
        @testset "$(typeof(A))" for A in
                                    matrix_versions(Symmetric(sprand(rng, Bool, n, n, p)))
            symmetric_color = symmetric_coloring(A, algo)
            @test check_symmetrically_orthogonal_columns(A, symmetric_color)
            @test minimum(symmetric_color) == 1
            @test maximum(symmetric_color) < size(A, 2) ÷ 2
        end
    end
end;
