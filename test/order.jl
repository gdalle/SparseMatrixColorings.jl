using SparseArrays
using SparseMatrixColorings:
    Graph,
    adjacency_graph,
    bipartite_graph,
    LargestFirst,
    NaturalOrder,
    RandomOrder,
    vertices
using Test

@testset "NaturalOrder" begin
    A = sprand(rng, Bool, 5, 5, 0.5)
    ag = adjacency_graph(A)
    @test vertices(ag, NaturalOrder()) == 1:5

    A = sprand(rng, Bool, 5, 4, 0.5)
    bg = bipartite_graph(A)
    @test vertices(bg, Val(1), NaturalOrder()) == 1:5

    A = sprand(rng, Bool, 5, 4, 0.5)
    bg = bipartite_graph(A)
    @test vertices(bg, Val(2), NaturalOrder()) == 1:4
end;

@testset "RandomOrder" begin
    A = sprand(rng, Bool, 5, 5, 0.5)
    ag = adjacency_graph(A)
    @test sort(vertices(ag, RandomOrder())) == 1:5

    A = sprand(rng, Bool, 5, 4, 0.5)
    bg = bipartite_graph(A)
    @test sort(vertices(bg, Val(1), RandomOrder())) == 1:5

    A = sprand(rng, Bool, 5, 4, 0.5)
    bg = bipartite_graph(A)
    @test sort(vertices(bg, Val(2), RandomOrder())) == 1:4
end;

@testset "LargestFirst" begin
    A = sparse([
        0 1 0
        1 0 0
        0 1 0
    ])
    ag = adjacency_graph(A)

    @test vertices(ag, LargestFirst()) == [2, 1, 3]

    A = sparse([
        1 1 1
        1 0 0
        0 1 1
        0 0 0
    ])
    bg = bipartite_graph(A)

    @test vertices(bg, Val(1), LargestFirst()) == [1, 3, 2, 4]

    A = sparse([
        1 1 0 0
        1 0 1 0
        1 0 1 0
    ])
    bg = bipartite_graph(A)

    @test vertices(bg, Val(2), LargestFirst()) == [1, 3, 2, 4]
end;
