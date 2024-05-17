using LinearAlgebra
using SparseArrays
using SparseMatrixColorings: col_major, row_major, nz_in_col, nz_in_row
using Test

A_dense_colmajor = [
    1 0 0 0
    1 1 0 0
    0 1 1 0
]
A_dense_rowmajor = transpose(Matrix(transpose(A_dense_colmajor)))

A_sparse_colmajor = sparse(A_dense_colmajor)
A_sparse_rowmajor = transpose(sparse(transpose(A_dense_colmajor)))

@testset "$(typeof(A))" for A in (
    A_dense_colmajor, A_dense_rowmajor, A_sparse_colmajor, A_sparse_rowmajor
)
    C = col_major(A)
    R = row_major(A)

    @test C == A
    @test R == A

    @test nz_in_col(C, 1) == nz_in_col(R, 1) == [1, 2]
    @test nz_in_col(C, 2) == nz_in_col(R, 2) == [2, 3]
    @test nz_in_col(C, 3) == nz_in_col(R, 3) == [3]
    @test nz_in_col(C, 4) == nz_in_col(R, 4) == Int[]
    @test nz_in_row(C, 1) == nz_in_row(R, 1) == [1]
    @test nz_in_row(C, 2) == nz_in_row(R, 2) == [1, 2]
    @test nz_in_row(C, 3) == nz_in_row(R, 3) == [2, 3]
end
