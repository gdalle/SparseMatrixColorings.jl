using ADTypes: column_coloring, row_coloring, symmetric_coloring
using JET
using LinearAlgebra
using SparseArrays
using SparseMatrixColorings
using SparseMatrixColorings: matrix_versions, respectful_similar
using StableRNGs
using Test

rng = StableRNG(63)

@testset "Sparse coloring" begin
    n = 10
    A = sprand(rng, n, n, 5 / n)

    # ADTypes
    @testset "ADTypes" begin
        @test_opt target_modules = (SparseMatrixColorings,) column_coloring(
            A, GreedyColoringAlgorithm()
        )
        @test_opt target_modules = (SparseMatrixColorings,) row_coloring(
            A, GreedyColoringAlgorithm()
        )
        @test_opt target_modules = (SparseMatrixColorings,) symmetric_coloring(
            Symmetric(A), GreedyColoringAlgorithm()
        )
    end

    @testset "$structure - $partition - $decompression" for (
        structure, partition, decompression
    ) in [
        (:nonsymmetric, :column, :direct),
        (:nonsymmetric, :row, :direct),
        (:symmetric, :column, :direct),
        (:symmetric, :column, :substitution),
    ]
        @test_opt target_modules = (SparseMatrixColorings,) coloring(
            A,
            ColoringProblem(; structure, partition),
            GreedyColoringAlgorithm(; decompression),
        )
    end
end;

@testset "Structured coloring" begin
    n = 10
    @testset "$(nameof(typeof(A))) - $structure - $partition - $decompression" for A in [
            Diagonal(rand(n)),
            Bidiagonal(rand(n), rand(n - 1), 'U'),
            Bidiagonal(rand(n), rand(n - 1), 'L'),
            Tridiagonal(rand(n - 1), rand(n), rand(n - 1)),
        ],
        (structure, partition, decompression) in
        [(:nonsymmetric, :column, :direct), (:nonsymmetric, :row, :direct)]

        @test_opt target_modules = (SparseMatrixColorings,) coloring(
            A,
            ColoringProblem(; structure, partition),
            GreedyColoringAlgorithm(; decompression),
        )
    end
end;

@testset "Sparse decompression" begin
    n = 10
    A0 = sparse(Symmetric(sprand(rng, n, n, 5 / n)))

    @testset "$structure - $partition - $decompression" for (
        structure, partition, decompression
    ) in [
        (:nonsymmetric, :column, :direct),
        (:nonsymmetric, :row, :direct),
        (:symmetric, :column, :direct),
        (:symmetric, :column, :substitution),
    ]
        @testset "A::$(typeof(A))" for A in matrix_versions(A0)
            result = coloring(
                A0,
                ColoringProblem(; structure, partition),
                GreedyColoringAlgorithm(; decompression);
                decompression_eltype=eltype(A),
            )
            B = compress(A, result)
            @testset "Full decompression" begin
                @test_opt compress(A, result)
                @test_opt decompress(B, result) ≈ A0
                @test_opt decompress!(respectful_similar(A), B, result)
            end
            @testset "Single-color decompression" begin
                if decompression == :direct
                    b = if partition == :column
                        B[:, 1]
                    else
                        B[1, :]
                    end
                    @test_opt decompress_single_color!(respectful_similar(A), b, 1, result)
                end
            end
            @testset "Triangle decompression" begin
                if structure == :symmetric
                    @test_opt decompress!(respectful_similar(triu(A)), B, result, :U)
                end
            end
            @testset "Single-color triangle decompression" begin
                if structure == :symmetric && decompression == :direct
                    @test_opt decompress_single_color!(
                        respectful_similar(triu(A)), B[:, 1], 1, result, :U
                    )
                end
            end
        end
    end
end;

@testset "Structured decompression" begin
    n = 10
    @testset "$(nameof(typeof(A))) - $structure - $partition - $decompression" for A in [
            Diagonal(rand(n)),
            Bidiagonal(rand(n), rand(n - 1), 'U'),
            Bidiagonal(rand(n), rand(n - 1), 'L'),
            Tridiagonal(rand(n - 1), rand(n), rand(n - 1)),
        ],
        (structure, partition, decompression) in
        [(:nonsymmetric, :column, :direct), (:nonsymmetric, :row, :direct)]

        result = coloring(
            A,
            ColoringProblem(; structure, partition),
            GreedyColoringAlgorithm(; decompression);
        )
        B = compress(A, result)
        @test_opt decompress(B, result)
    end
end;
