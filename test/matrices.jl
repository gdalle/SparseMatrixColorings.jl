using SparseArrays
using SparseMatrixColorings: matrix_versions, respectful_similar, same_sparsity_pattern
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

@testset "Sparsity pattern comparison" begin
    A = [
        1 1
        0 1
        0 0
    ]
    B1 = [
        1 1
        0 1
        1 0
    ]
    B2 = [
        1 1
        0 0
        0 1
    ]
    C = [
        1 1 0
        0 1 0
        0 0 0
    ]

    @test same_sparsity_pattern(sparse(A), sparse(A))
    @test !same_sparsity_pattern(sparse(A), sparse(B1))
    @test_broken !same_sparsity_pattern(sparse(A), sparse(B2))
    @test !same_sparsity_pattern(sparse(A), sparse(C))

    @test same_sparsity_pattern(transpose(sparse(A)), transpose(sparse(A)))
    @test !same_sparsity_pattern(transpose(sparse(A)), transpose(sparse(B1)))
    @test_broken !same_sparsity_pattern(transpose(sparse(A)), transpose(sparse(B2)))
    @test !same_sparsity_pattern(transpose(sparse(A)), transpose(sparse(C)))
end;
