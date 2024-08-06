using Base.Iterators: product
using Compat
using LinearAlgebra
using SparseArrays
using SparseMatrixColorings
using SparseMatrixColorings:
    SimpleColoringResult,
    color_groups,
    decompress_columns,
    decompress_columns!,
    decompress_rows,
    decompress_rows!,
    decompress_symmetric,
    decompress_symmetric!,
    matrix_versions,
    same_sparsity_pattern
using StableRNGs
using Test

rng = StableRNG(63)

algo = GreedyColoringAlgorithm()

@testset "Column decompression" begin
    A0 = [
        1 0 2
        0 3 4
        5 0 0
    ]
    S0 = map(!iszero, A0)
    B = [
        1 2
        3 4
        5 0
    ]
    color = [1, 1, 2]
    @testset "A::$(typeof(A)) - S::$(typeof(S))" for (A, S) in product(
        matrix_versions(A0), matrix_versions(S0)
    )
        @test decompress_columns(S, B, SimpleColoringResult(color)) == A
    end
end;

@testset "Row decompression" begin
    A0 = [
        1 0 3
        0 2 0
        4 5 0
    ]
    S0 = map(!iszero, A0)
    B = [
        1 2 3
        4 5 0
    ]
    color = [1, 1, 2]
    @testset "A::$(typeof(A)) - S::$(typeof(S))" for (A, S) in product(
        matrix_versions(A0), matrix_versions(S0)
    )
        @test decompress_rows(S, B, SimpleColoringResult(color)) == A
    end
end;

@testset "Symmetric decompression" begin
    @testset "Fig 4.1 from 'What color is your Jacobian'" begin
        A0 = what_fig_41()
        S0 = map(!iszero, A0)
        color = [
            1,  # green
            2,  # red
            1,  # green
            3,  # blue
            1,  # green
            1,  # green
        ]
        group = color_groups(color)
        B = stack(group; dims=2) do g
            dropdims(sum(A0[:, g]; dims=2); dims=2)
        end
        @testset "A::$(typeof(A)) - S::$(typeof(S))" for (A, S) in product(
            matrix_versions(A0), matrix_versions(S0)
        )
            @test decompress_symmetric(S, B, SimpleColoringResult(color)) == A
        end
    end

    @testset "Fig 1 from 'Efficient computation of sparse hessians using coloring and AD'" begin
        A0 = efficient_fig_1()
        S0 = map(!iszero, A0)
        color = [
            1,  # red
            2,  # blue
            1,  # red
            3,  # yellow
            1,  # red
            4,  # green
            3,  # yellow
            5,  # navy blue
            1,  # red
            2,  # blue
        ]
        group = color_groups(color)
        B = stack(group; dims=2) do g
            dropdims(sum(A0[:, g]; dims=2); dims=2)
        end
        #! format: off
        B_th = [
                            A0[1,1]    A0[1,2]  A0[1,7]        0        0
            A0[2,1]+A0[2,3]+A0[2,5]    A0[2,2]        0        0        0
                            A0[3,3]    A0[3,2]  A0[3,4]  A0[3,6]        0
                            A0[4,3]   A0[4,10]  A0[4,4]        0        0
                            A0[5,5]    A0[5,2]        0  A0[5,6]  A0[5,8]
            A0[6,3]+A0[6,5]+A0[6,9]          0        0  A0[6,6]        0
                            A0[7,1]          0  A0[7,7]        0  A0[7,8]
                   A0[8,5]+A0[8,9]           0  A0[8,7]        0  A0[8,8]
                            A0[9,9]   A0[9,10]        0  A0[9,6]  A0[9,8]
                           A0[10,9]  A0[10,10] A0[10,4]        0        0
        ]
        @test B == B_th
        #! format: on
        @testset "A::$(typeof(A)) - S::$(typeof(S))" for (A, S) in product(
            matrix_versions(A0), matrix_versions(S0)
        )
            @test decompress_symmetric(S, B, SimpleColoringResult(color)) == A
        end
    end
end;
