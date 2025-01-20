using LinearAlgebra
using SparseArrays
using SparseMatrixColorings
using SparseMatrixColorings:
    AdjacencyGraph,
    BipartiteGraph,
    LargestFirst,
    NaturalOrder,
    RandomOrder,
    degree_dist2,
    nb_vertices,
    valid_dynamic_order,
    vertices
using Random
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
    A = sprand(rng, Bool, 10, 10, 0.5)
    ag = AdjacencyGraph(A)
    @test sort(vertices(ag, RandomOrder(rng))) == 1:10
    @test sort(vertices(ag, RandomOrder())) == 1:10

    A = sprand(rng, Bool, 10, 8, 0.5)
    bg = BipartiteGraph(A)
    @test sort(vertices(bg, Val(1), RandomOrder(rng))) == 1:10
    @test sort(vertices(bg, Val(1), RandomOrder())) == 1:10

    A = sprand(rng, Bool, 10, 8, 0.5)
    bg = BipartiteGraph(A)
    @test sort(vertices(bg, Val(2), RandomOrder(rng))) == 1:8
    @test sort(vertices(bg, Val(2), RandomOrder())) == 1:8

    order = RandomOrder()
    @test order.rng === Random.default_rng()
    @test isnothing(order.seed)

    order = RandomOrder(StableRNG(0), 6)
    @test order.seed == 6
    @test vertices(ag, order) == vertices(ag, order)
    @test vertices(bg, Val(2), order) == vertices(bg, Val(2), order)
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

@testset "Dynamic degree-based orders" begin
    @testset "$order" for order in
                          [SmallestLast(), IncidenceDegree(), DynamicLargestFirst()]
        @testset "AdjacencyGraph" begin
            for (n, p) in Iterators.product(20:20:100, 0.0:0.1:0.2)
                yield()
                A = sparse(Symmetric(sprand(rng, Bool, n, n, p)))
                g = AdjacencyGraph(A)
                π = vertices(g, order)
                @test valid_dynamic_order(g, π, order)
            end
        end
        @testset "BipartiteGraph" begin
            for (n, p) in Iterators.product(20:20:100, 0.0:0.1:0.2)
                m = rand((n ÷ 2, n * 2))
                A = sprand(rng, Bool, m, n, p)
                g = BipartiteGraph(A)
                π1 = vertices(g, Val(1), order)
                π2 = vertices(g, Val(2), order)
                @test valid_dynamic_order(g, Val(1), π1, order)
                @test valid_dynamic_order(g, Val(2), π2, order)
                @test !valid_dynamic_order(g, Val(1), π2, order)
            end
        end
    end
end;
