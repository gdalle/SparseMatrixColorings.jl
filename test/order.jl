using SparseArrays
using SparseMatrixColorings:
    BipartiteGraph,
    AdjacencyGraph,
    BipartiteGraph,
    LargestFirst,
    NaturalOrder,
    RandomOrder,
    degree_dist2,
    vertices
using StableRNGs
using Test

rng = StableRNG(63)

@testset "NaturalOrder" begin
    A = sprand(rng, Bool, 5, 5, 0.5)
    ag = AdjacencyGraph(A)
    @test vertices(ag, NaturalOrder()) == 1:5

    A = sprand(rng, Bool, 5, 4, 0.5)
    bg = BipartiteGraph(A)
    @test vertices(bg, Val(1), NaturalOrder()) == 1:5

    A = sprand(rng, Bool, 5, 4, 0.5)
    bg = BipartiteGraph(A)
    @test vertices(bg, Val(2), NaturalOrder()) == 1:4
end;

@testset "RandomOrder" begin
    A = sprand(rng, Bool, 5, 5, 0.5)
    ag = AdjacencyGraph(A)
    @test sort(vertices(ag, RandomOrder(rng))) == 1:5
    @test sort(vertices(ag, RandomOrder())) == 1:5

    A = sprand(rng, Bool, 5, 4, 0.5)
    bg = BipartiteGraph(A)
    @test sort(vertices(bg, Val(1), RandomOrder(rng))) == 1:5
    @test sort(vertices(bg, Val(1), RandomOrder())) == 1:5

    A = sprand(rng, Bool, 5, 4, 0.5)
    bg = BipartiteGraph(A)
    @test sort(vertices(bg, Val(2), RandomOrder(rng))) == 1:4
    @test sort(vertices(bg, Val(2), RandomOrder())) == 1:4
end;

@testset "LargestFirst" begin
    A = sparse([
        0 1 0
        1 0 0
        0 1 0
    ])
    ag = AdjacencyGraph(A)

    @test vertices(ag, LargestFirst()) == [2, 1, 3]

    A = sparse([
        1 1 0 0
        0 1 1 1
        0 0 1 0
        0 0 0 0
        1 0 1 0
    ])
    bg = BipartiteGraph(A)

    for side in (1, 2)
        true_order = sort(
            vertices(bg, Val(side)); by=v -> degree_dist2(bg, Val(side), v), rev=true
        )
        @test vertices(bg, Val(side), LargestFirst()) == true_order
    end
end;
