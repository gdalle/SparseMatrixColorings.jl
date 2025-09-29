using ArrayInterface: ArrayInterface
using BandedMatrices: BandedMatrix, brand
using BlockBandedMatrices: BandedBlockBandedMatrix, BlockBandedMatrix
using LinearAlgebra
using SparseMatrixColorings
using Test

@testset "Diagonal" begin
    @testset for algo in [GreedyColoringAlgorithm(), StructuredColoringAlgorithm()]
        for n in (1, 2, 10, 100)
            A = Diagonal(rand(n))
            test_structured_coloring_decompression(A, algo)
        end
    end
end;

@testset "Bidiagonal" begin
    @testset for algo in [GreedyColoringAlgorithm(), StructuredColoringAlgorithm()]
        for n in (2, 10, 100)
            A1 = Bidiagonal(rand(n), rand(n - 1), :U)
            A2 = Bidiagonal(rand(n), rand(n - 1), :L)
            test_structured_coloring_decompression(A1, algo)
            test_structured_coloring_decompression(A2, algo)
        end
    end
end;

@testset "Tridiagonal" begin
    @testset for algo in [GreedyColoringAlgorithm(), StructuredColoringAlgorithm()]
        for n in (2, 10, 100)
            A = Tridiagonal(rand(n - 1), rand(n), rand(n - 1))
            test_structured_coloring_decompression(A, algo)
        end
    end
end;

@testset "BandedMatrices" begin
    @testset for algo in [GreedyColoringAlgorithm(), StructuredColoringAlgorithm()]
        @testset for (m, n) in [(10, 20), (20, 10)], l in 0:5, u in 0:5
            A = brand(m, n, l, u)
            test_structured_coloring_decompression(A, algo)
        end
    end
end;

@testset "BlockBandedMatrices" begin
    @testset for algo in [GreedyColoringAlgorithm(), StructuredColoringAlgorithm()]
        for (mb, nb) in [(10, 20), (20, 10)], lb in 0:3, ub in 0:3, _ in 1:10
            rows = rand(1:5, mb)
            cols = rand(1:5, nb)
            A = BlockBandedMatrix{Float64}(rand(sum(rows), sum(cols)), rows, cols, (lb, ub))
            test_structured_coloring_decompression(A, algo)
        end
    end
end;

@testset "BandedBlockBandedMatrices" begin
    @testset for algo in [GreedyColoringAlgorithm(), StructuredColoringAlgorithm()]
        for (mb, nb) in [(10, 20), (20, 10)], lb in 0:3, ub in 0:3, _ in 1:10
            rows = rand(5:10, mb)
            cols = rand(5:10, nb)
            λ = rand(0:5)
            μ = rand(0:5)
            A = BandedBlockBandedMatrix{Float64}(
                rand(sum(rows), sum(cols)), rows, cols, (lb, ub), (λ, μ)
            )
            test_structured_coloring_decompression(A, algo)
        end
    end
end;
