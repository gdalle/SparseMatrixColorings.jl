using ADTypes: column_coloring, row_coloring, symmetric_coloring
using JET
using LinearAlgebra
using SparseArrays
using SparseMatrixColorings
using SparseMatrixColorings: matrix_versions, respectful_similar, all_orders
using StableRNGs
using Test

rng = StableRNG(63)

@testset "Sparse coloring" begin
    n = 10
    A = sparse(Symmetric(sprand(rng, n, n, 5 / n)))

    # ADTypes
    @testset "ADTypes" begin
        @test_opt column_coloring(A, GreedyColoringAlgorithm())
        @test_opt row_coloring(A, GreedyColoringAlgorithm())
        @test_opt symmetric_coloring(Symmetric(A), GreedyColoringAlgorithm())

        @inferred column_coloring(A, GreedyColoringAlgorithm())
        @inferred row_coloring(A, GreedyColoringAlgorithm())
        @inferred symmetric_coloring(Symmetric(A), GreedyColoringAlgorithm())
    end

    @testset "$structure - $partition - $decompression" for (
        structure, partition, decompression
    ) in [
        (:nonsymmetric, :column, :direct),
        (:nonsymmetric, :row, :direct),
        (:symmetric, :column, :direct),
        (:symmetric, :column, :substitution),
        (:nonsymmetric, :bidirectional, :direct),
        (:nonsymmetric, :bidirectional, :substitution),
    ]
        @test_opt coloring(
            A,
            ColoringProblem(; structure, partition),
            GreedyColoringAlgorithm(; decompression),
        )
        @inferred coloring(
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

        @test_opt coloring(
            A,
            ColoringProblem(; structure, partition),
            GreedyColoringAlgorithm(; decompression),
        )
        @inferred coloring(
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
        (:nonsymmetric, :bidirectional, :direct),
        (:nonsymmetric, :bidirectional, :substitution),
    ]
        @testset "A::$(typeof(A))" for A in matrix_versions(A0)
            result = coloring(
                A0,
                ColoringProblem(; structure, partition),
                GreedyColoringAlgorithm(; decompression);
                decompression_eltype=eltype(A),
            )
            if partition == :bidirectional
                Br, Bc = compress(A, result)
                @testset "Full decompression" begin
                    @test_opt compress(A, result)
                    @test_opt decompress(Br, Bc, result)
                    @test_opt decompress!(respectful_similar(A), Br, Bc, result)

                    @inferred compress(A, result)
                    @inferred decompress(Br, Bc, result)
                end
            else
                B = compress(A, result)
                @testset "Full decompression" begin
                    @test_opt compress(A, result)
                    @test_opt decompress(B, result)
                    @test_opt decompress!(respectful_similar(A), B, result)

                    @inferred compress(A, result)
                    @inferred decompress(B, result)
                end
                @testset "Single-color decompression" begin
                    if decompression == :direct
                        b = if partition == :column
                            B[:, 1]
                        else
                            B[1, :]
                        end
                        @test_opt decompress_single_color!(
                            respectful_similar(A), b, 1, result
                        )
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
        @inferred decompress(B, result)
    end
end;

@testset "Single precision" begin
    A = convert(
        SparseMatrixCSC{Float32,Int32},
        sparse(Symmetric(sprand(rng, Float32, 100, 100, 0.1))),
    )
    @testset "$structure - $partition - $decompression" for (
        structure, partition, decompression
    ) in [
        (:nonsymmetric, :column, :direct),
        (:nonsymmetric, :row, :direct),
        (:symmetric, :column, :direct),
        (:symmetric, :column, :substitution),
        (:nonsymmetric, :bidirectional, :direct),
        (:nonsymmetric, :bidirectional, :substitution),
    ]
        @testset for order in all_orders()
            result = coloring(
                A,
                ColoringProblem(; structure, partition),
                GreedyColoringAlgorithm(order; decompression);
            )
            if partition in (:column, :bidirectional)
                @test eltype(column_colors(result)) == Int32
                @test eltype(column_groups(result)[1]) == Int32
            end
            if partition in (:row, :bidirectional)
                @test eltype(row_colors(result)) == Int32
                @test eltype(row_groups(result)[1]) == Int32
            end
            if partition == :bidirectional
                Br, Bc = compress(A, result)
                @test decompress(Br, Bc, result) isa SparseMatrixCSC{Float32,Int32}
            else
                B = compress(A, result)
                @test decompress(B, result) isa SparseMatrixCSC{Float32,Int32}
            end
        end
    end
end
