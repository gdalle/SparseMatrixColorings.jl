using LinearAlgebra
using SparseArrays
using SparseMatrixColorings:
    check_same_pattern, matrix_versions, respectful_similar, same_pattern
using StableRNGs
using Test

rng = StableRNG(63)

@testset "Matrix versions" begin
    A0_dense = rand(3, 4)
    A0_sparse = sprand(rng, 10, 20, 0.3)
    @test all(==(A0_dense), matrix_versions(A0_dense))
    @test all(==(A0_sparse), matrix_versions(A0_sparse))
end

same_view(::AbstractMatrix, ::AbstractMatrix) = false
same_view(::Matrix, ::Matrix) = true
same_view(::SparseMatrixCSC, ::SparseMatrixCSC) = true
same_view(::Transpose, ::Transpose) = true
same_view(::Adjoint, ::Adjoint) = true

@testset "Respectful similar" begin
    A0_dense = rand(3, 4)
    A0_sparse = sprand(rng, 10, 20, 0.3)
    @test all(matrix_versions(A0_dense)) do A
        B = respectful_similar(A)
        size(B) == size(A) && same_view(A, B)
    end
    @test all(matrix_versions(A0_sparse)) do A
        B = respectful_similar(A)
        size(B) == size(A) && same_view(A, B)
    end
end

@testset "Sparsity pattern" begin
    S = sparse([
        0 1 1
        0 1 0
        1 1 0
    ])

    A1 = copy(S)
    A2 = copy(S)
    A2[1, 1] = 1

    @test same_pattern(A1, S)
    @test !same_pattern(A2, S)
    @test same_pattern(Matrix(A2), S)

    @test_throws DimensionMismatch check_same_pattern(vcat(A1, A1), S)
    @test_throws DimensionMismatch check_same_pattern(A2, S)
    @test check_same_pattern(A1, S; allow_denser=true)
    @test check_same_pattern(A2, S; allow_denser=true)
end
