using SparseArrays
using SparseMatrixColorings
using SparseMatrixColorings:
    structurally_orthogonal_columns,
    symmetrically_orthogonal_columns,
    directly_recoverable_columns,
    what_fig_41,
    what_fig_61,
    efficient_fig_1,
    efficient_fig_4
using Test

@testset "Column coloring & decompression" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)
    algo = GreedyColoringAlgorithm(; decompression=:direct)
    A0 = sparse([
        1 0 2
        0 3 4
        5 0 0
    ])
    B0 = [
        1 2
        3 4
        5 0
    ]
    color0 = [1, 1, 2]
    @test structurally_orthogonal_columns(A0, color0)
    @test directly_recoverable_columns(A0, color0)
    test_coloring_decompression(A0, problem, algo; B0, color0)
end;

@testset "Row coloring & decompression" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:row)
    algo = GreedyColoringAlgorithm(; decompression=:direct)
    A0 = sparse([
        1 0 3
        0 2 0
        4 5 0
    ])
    B0 = [
        1 2 3
        4 5 0
    ]
    color0 = [1, 1, 2]
    @test structurally_orthogonal_columns(transpose(A0), color0)
    @test directly_recoverable_columns(transpose(A0), color0)
    test_coloring_decompression(A0, problem, algo; B0, color0)
end;

@testset "Symmetric coloring & direct decompression" begin
    problem = ColoringProblem(; structure=:symmetric, partition=:column)
    algo = GreedyColoringAlgorithm(; decompression=:direct)

    @testset "Fig 4.1 from 'What color is your Jacobian'" begin
        example = what_fig_41()
        A0, B0, color0 = example.A, example.B, example.color
        @test symmetrically_orthogonal_columns(A0, color0)
        @test directly_recoverable_columns(A0, color0)
        test_coloring_decompression(A0, problem, algo; B0, color0)
    end

    @testset "Fig 1 from 'Efficient computation of sparse hessians using coloring and AD'" begin
        example = efficient_fig_1()
        A0, B0, color0 = example.A, example.B, example.color
        @test symmetrically_orthogonal_columns(A0, color0)
        @test directly_recoverable_columns(A0, color0)
        test_coloring_decompression(A0, problem, algo; B0, color0)
    end
end;

@testset "Symmetric coloring & substitution decompression" begin
    problem = ColoringProblem(; structure=:symmetric, partition=:column)
    algo = GreedyColoringAlgorithm(; decompression=:substitution)

    @testset "Fig 6.1 from 'What color is your Jacobian'" begin
        example = what_fig_61()
        A0, B0, color0 = example.A, example.B, example.color
        # our coloring doesn't give the color0 from the example, but that's okay
        test_coloring_decompression(A0, problem, algo)
    end

    @testset "Fig 4 from 'Efficient computation of sparse hessians using coloring and AD'" begin
        example = efficient_fig_4()
        A0, B0, color0 = example.A, example.B, example.color
        test_coloring_decompression(A0, problem, algo; B0, color0)
    end
end;
