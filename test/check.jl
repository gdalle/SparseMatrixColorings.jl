using SparseMatrixColorings:
    check_structurally_orthogonal_columns,
    check_structurally_orthogonal_rows,
    check_symmetrically_orthogonal
using Test

@testset "Column coloring" begin
    A = [
        1 0 0
        0 1 0
        0 1 1
    ]
    @test check_structurally_orthogonal_columns(A, [1, 2, 3])
    @test check_structurally_orthogonal_columns(A, [1, 2, 1])
    @test check_structurally_orthogonal_columns(A, [1, 1, 2])
    @test !check_structurally_orthogonal_columns(A, [1, 2, 2])
end

@testset "Row coloring" begin
    A = [
        1 0 0
        0 1 0
        0 1 1
    ]
    @test check_structurally_orthogonal_rows(A, [1, 2, 3])
    @test check_structurally_orthogonal_rows(A, [1, 2, 1])
    @test check_structurally_orthogonal_rows(A, [1, 1, 2])
    @test !check_structurally_orthogonal_rows(A, [1, 2, 2])
end

@testset "Symmetric coloring" begin
    # example from "What color is your Jacobian", fig 4.1
    A = [
        1 1 0 0 0 0
        1 1 1 0 1 1
        0 1 1 1 0 0
        0 0 1 1 0 1
        0 1 0 0 1 0
        0 1 0 1 0 1
    ]
    @test check_symmetrically_orthogonal(A, [1, 2, 1, 3, 1, 1])
    @test !check_symmetrically_orthogonal(A, [1, 3, 1, 3, 1, 1])
end
