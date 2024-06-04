using Chairmarks
using ColPack: ColPack
using LinearAlgebra
using SparseMatrixColorings
using SparseMatrixColorings: NaturalOrder
using SparseArrays
using StableRNGs
using Test

rng = StableRNG(63)

function my_symmetric_coloring(H)
    return symmetric_coloring(H, GreedyColoringAlgorithm(NaturalOrder()))
end

function colpack_symmetric_coloring(H)
    method = ColPack.star_coloring()
    ordering = ColPack.natural_ordering()
    coloring = ColPack.ColPackColoring(H, method, ordering)
    return ColPack.get_colors(coloring)
end

n_values = floor.(Int, 10 .^ (1:5))
p_values(n) = (2:5:min(n, 30)) ./ n

@testset verbose = true "Star coloring" begin
    @testset "Correctness" begin
        @testset "n=$n - p=$p" for n in n_values, p in p_values(n)
            H = sparse(Symmetric(sprand(rng, n, n, p)))
            color1 = my_symmetric_coloring(H)
            color2 = colpack_symmetric_coloring(H)
            @test color1 == color2
        end
    end
    @testset "Performance" begin
        @testset "n=$n - p=$p" for n in n_values, p in p_values(n)
            bench1 = @be sparse(Symmetric(sprand(rng, n, n, p))) my_symmetric_coloring
            bench2 = @be sparse(Symmetric(sprand(rng, n, n, p))) colpack_symmetric_coloring
            time1 = minimum(bench1).time
            time2 = minimum(bench2).time
            @test time2 > time1
        end
    end
end;
