using Base.Iterators: product
using Compat
using LinearAlgebra
using SparseArrays
using SparseMatrixColorings
using SparseMatrixColorings:
    DefaultColoringResult,
    decompress,
    decompress!,
    group_by_color,
    matrix_versions,
    respectful_similar
using Test

algo = GreedyColoringAlgorithm()

@testset "Column decompression" begin
    A0 = sparse([
        1 0 2
        0 3 4
        5 0 0
    ])
    S0 = map(!iszero, A0)
    B = [
        1 2
        3 4
        5 0
    ]
    color = [1, 1, 2]
    result = DefaultColoringResult{:nonsymmetric,:column,:direct}(A0, color)
    @test decompress(B, result) == A0
    for A in matrix_versions(A0)
        @test decompress!(respectful_similar(A), B, result) == A
    end
end;

@testset "Row decompression" begin
    A0 = sparse([
        1 0 3
        0 2 0
        4 5 0
    ])
    B = [
        1 2 3
        4 5 0
    ]
    color = [1, 1, 2]
    result = DefaultColoringResult{:nonsymmetric,:row,:direct}(A0, color)
    @test decompress(B, result) == A0
    for A in matrix_versions(A0)
        @test decompress!(respectful_similar(A), B, result) == A
    end
end;

@testset "Symmetric decompression" begin
    @testset "Fig 4.1 from 'What color is your Jacobian'" begin
        A0 = what_fig_41()
        color = [
            1,  # green
            2,  # red
            1,  # green
            3,  # blue
            1,  # green
            1,  # green
        ]
        result = DefaultColoringResult{:symmetric,:column,:direct}(A0, color)
        group = group_by_color(color)
        B = stack(group; dims=2) do g
            dropdims(sum(A0[:, g]; dims=2); dims=2)
        end
        @test decompress(B, result) == A0
        for A in matrix_versions(A0)
            @test decompress!(respectful_similar(A), B, result) == A
        end
    end

    @testset "Fig 1 from 'Efficient computation of sparse hessians using coloring and AD'" begin
        A0 = efficient_fig_1()
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
        result = DefaultColoringResult{:symmetric,:column,:direct}(A0, color)
        group = group_by_color(color)
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
        @test decompress(B, result) == A0
        for A in matrix_versions(A0)
            @test decompress!(respectful_similar(A), B, result) == A
        end
    end
end;
