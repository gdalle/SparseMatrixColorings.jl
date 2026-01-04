using LinearAlgebra
using SparseArrays
using SparseMatrixColorings:
    BipartiteGraph,
    AdjacencyGraph,
    matrix_versions,
    respectful_similar,
    compatible_pattern,
    check_compatible_pattern
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

@testset "Compatible sparsity pattern -- BipartiteGraph" begin
    S = sparse([
        0 1 1
        0 1 0
        1 1 0
    ])

    A1 = copy(S)
    A2 = copy(S)
    A2[1, 1] = 1

    bg1 = BipartiteGraph(A1)
    bg2 = BipartiteGraph(A2)
    @test compatible_pattern(S, bg1)
    @test !compatible_pattern(S, bg2)
    @test compatible_pattern(Matrix(S), bg1)

    @test_throws DimensionMismatch check_compatible_pattern(S, bg2)
end

@testset "Compatible sparsity pattern -- AdjacencyGraph" begin
    S = sparse([
        1 0 1
        0 1 1
        1 1 0
    ])

    A1 = copy(S)
    A2 = copy(S)
    A2[3, 3] = 1

    ag1 = AdjacencyGraph(A1)
    ag2 = AdjacencyGraph(A2)
    for (op, uplo) in ((tril, :L), (triu, :U), (identity, :F))
        @test compatible_pattern(op(S), ag1, uplo)
        @test !compatible_pattern(op(S), ag2, uplo)
        @test compatible_pattern(Matrix(op(S)), ag1, uplo)

        @test_throws DimensionMismatch check_compatible_pattern(op(S), ag2, uplo)
    end
end
