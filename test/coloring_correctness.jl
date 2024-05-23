using ADTypes: column_coloring, row_coloring, symmetric_coloring
using LinearAlgebra: I, Symmetric
using SparseArrays: sprand
using SparseMatrixColorings:
    GreedyColoringAlgorithm,
    check_structurally_orthogonal_columns,
    check_structurally_orthogonal_rows,
    check_symmetrically_orthogonal
using StableRNGs
using Test

rng = StableRNG(63)

algo = GreedyColoringAlgorithm()

@test startswith(string(algo), "GreedyColoringAlgorithm(")

@testset "Column coloring" begin
    @testset "$(typeof(A)) - $(size(A))" for A in (
        sprand(rng, Bool, 100, 200, 0.05),
        sprand(rng, Bool, 200, 100, 0.05),
        Matrix(sprand(rng, Bool, 100, 200, 0.05)),
        Matrix(sprand(rng, Bool, 200, 100, 0.05)),
    )
        column_colors = column_coloring(A, algo)
        @test check_structurally_orthogonal_columns(A, column_colors)
        @test minimum(column_colors) == 1
        @test maximum(column_colors) < size(A, 2) ÷ 2
    end
end

@testset "Row coloring" begin
    @testset "$(typeof(A)) - $(size(A))" for A in (
        sprand(rng, Bool, 100, 200, 0.05),
        sprand(rng, Bool, 200, 100, 0.05),
        Matrix(sprand(rng, Bool, 100, 200, 0.05)),
        Matrix(sprand(rng, Bool, 200, 100, 0.05)),
    )
        row_colors = row_coloring(A, algo)
        @test check_structurally_orthogonal_rows(A, row_colors)
        @test minimum(row_colors) == 1
        @test maximum(row_colors) < size(A, 1) ÷ 2
    end
end

@testset "Symmetric coloring" begin
    @testset "$(typeof(A)) - $(size(A))" for A in (
        sparse(Symmetric(sprand(rng, Bool, 100, 100, 0.05))),
        Symmetric(Matrix(sprand(rng, Bool, 100, 100, 0.05))),
    )
        symmetric_colors = symmetric_coloring(A, algo)
        @test check_symmetrically_orthogonal(A, symmetric_colors)
        @test minimum(symmetric_colors) == 1
        @test maximum(symmetric_colors) < size(A, 2) ÷ 2
    end
end
