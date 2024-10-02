using ArrayInterface: ArrayInterface
using BandedMatrices: BandedMatrix, brand
using BlockBandedMatrices: BandedBlockBandedMatrix, BlockBandedMatrix
using LinearAlgebra
using SparseMatrixColorings
using SparseMatrixColorings: cycle_range
using Test

@testset "Utils" begin
    @test cycle_range(2, 3) == [1, 2, 1]
    @test cycle_range(2, 4) == [1, 2, 1, 2]
    @test cycle_range(2, 5) == [1, 2, 1, 2, 1]
    @test cycle_range(3, 5) == [1, 2, 3, 1, 2]
    @test cycle_range(3, 6) == [1, 2, 3, 1, 2, 3]
    @test cycle_range(2, 1) == [1]
    @test cycle_range(3, 1) == [1]
    @test cycle_range(3, 2) == [1, 2]
end;

@testset "Diagonal" begin
    for n in (1, 2, 10, 100)
        A = Diagonal(rand(n))
        test_structured_coloring_decompression(A)
    end
end;

@testset "Bidiagonal" begin
    for n in (2, 10, 100)
        A1 = Bidiagonal(rand(n), rand(n - 1), :U)
        A2 = Bidiagonal(rand(n), rand(n - 1), :L)
        test_structured_coloring_decompression(A1)
        test_structured_coloring_decompression(A2)
    end
end;

@testset "Tridiagonal" begin
    for n in (2, 10, 100)
        A = Tridiagonal(rand(n - 1), rand(n), rand(n - 1))
        test_structured_coloring_decompression(A)
    end
end;

@testset "BandedMatrices" begin
    @testset for (m, n) in [(10, 20), (20, 10)], l in 0:5, u in 0:5
        A = brand(m, n, l, u)
        test_structured_coloring_decompression(A)
    end
end;

@testset "BlockBandedMatrices" begin
    for (mb, nb) in [(10, 20), (20, 10)], lb in 0:3, ub in 0:3, _ in 1:10
        rows = rand(1:5, mb)
        cols = rand(1:5, nb)
        A = BlockBandedMatrix{Float64}(rand(sum(rows), sum(cols)), rows, cols, (lb, ub))
        test_structured_coloring_decompression(A)
    end
end;

@testset "BandedBlockBandedMatrices" begin
    for (mb, nb) in [(10, 20), (20, 10)], lb in 0:3, ub in 0:3, _ in 1:10
        rows = rand(5:10, mb)
        cols = rand(5:10, nb)
        λ = rand(0:5)
        μ = rand(0:5)
        A = BandedBlockBandedMatrix{Float64}(
            rand(sum(rows), sum(cols)), rows, cols, (lb, ub), (λ, μ)
        )
        test_structured_coloring_decompression(A)
    end
end;
