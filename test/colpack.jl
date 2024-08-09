using ADTypes: column_coloring, row_coloring, symmetric_coloring
using ColPack: ColPack, ColPackColoring
using LinearAlgebra
using SparseMatrixColorings
using SparseMatrixColorings: NaturalOrder
using SparseArrays
using StableRNGs
using Test

rng = StableRNG(63)

my_column_coloring(J) = column_coloring(J, GreedyColoringAlgorithm(NaturalOrder()))
my_row_coloring(J) = row_coloring(J, GreedyColoringAlgorithm(NaturalOrder()))
my_symmetric_coloring(H) = symmetric_coloring(H, GreedyColoringAlgorithm(NaturalOrder()))

function colpack_column_coloring(J)
    A = ColPack.matrix2adjmatrix(J; partition_by_rows=false)
    coloring = ColPackColoring(A, ColPack.d1_coloring(), ColPack.natural_ordering())
    return ColPack.get_colors(coloring)
end

function colpack_row_coloring(J)
    A = ColPack.matrix2adjmatrix(J; partition_by_rows=true)
    coloring = ColPackColoring(A, ColPack.d1_coloring(), ColPack.natural_ordering())
    return ColPack.get_colors(coloring)
end

function colpack_symmetric_coloring(H)
    coloring = ColPackColoring(H, ColPack.star_coloring(), ColPack.natural_ordering())
    return ColPack.get_colors(coloring)
end

n_values = floor.(Int, 10 .^ (1:3))
p_values(n) = (2:4:min(n, 20)) ./ n

@testset verbose = true "Correctness" begin
    @testset "Column coloring" begin
        @testset "n=$n - p=$p" for n in n_values, p in p_values(n)
            J = sprand(rng, n, n + 1, p)
            color1 = my_column_coloring(J)
            color2 = colpack_column_coloring(J)
            @test color1 == color2
        end
    end
    @testset "Row coloring" begin
        @testset "n=$n - p=$p" for n in n_values, p in p_values(n)
            J = sprand(rng, n, n + 1, p)
            color1 = my_row_coloring(J)
            color2 = colpack_row_coloring(J)
            @test color1 == color2
        end
    end
    @testset "Symmetric coloring" begin
        @testset "n=$n - p=$p" for n in n_values, p in p_values(n)
            H = sparse(Symmetric(sprand(rng, n, n, p)))
            color1 = my_symmetric_coloring(H)
            color2 = colpack_symmetric_coloring(H)
            @test color1 == color2
        end
    end
end;
