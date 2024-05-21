using SparseArrays
using SparseMatrixColorings: Graph, BipartiteGraph, AdjacencyGraph, neighbors
using Test

## Standard graph

@testset "Graph" begin
    g = Graph(sparse([
        1 0 1
        1 1 0
        0 0 0
    ]))

    @test length(g) == 3
    @test neighbors(g, 1) == [1, 2]
    @test neighbors(g, 2) == [2]
    @test neighbors(g, 3) == [1]
end;

## Bipartite graph (fig 3.1 of "What color is your Jacobian?")

@testset "BipartiteGraph" begin
    A = sparse([
        1 0 0 0 0 1 1 1
        0 1 0 0 1 0 1 1
        0 0 1 0 1 1 0 1
        0 0 0 1 1 1 1 0
    ])

    bg = BipartiteGraph(A)
    @test length(bg, Val(1)) == 4
    @test length(bg, Val(2)) == 8
    # neighbors of rows
    @test neighbors(bg, Val(1), 1) == [1, 6, 7, 8]
    @test neighbors(bg, Val(1), 2) == [2, 5, 7, 8]
    @test neighbors(bg, Val(1), 3) == [3, 5, 6, 8]
    @test neighbors(bg, Val(1), 4) == [4, 5, 6, 7]
    # neighbors of columns
    @test neighbors(bg, Val(2), 1) == [1]
    @test neighbors(bg, Val(2), 2) == [2]
    @test neighbors(bg, Val(2), 3) == [3]
    @test neighbors(bg, Val(2), 4) == [4]
    @test neighbors(bg, Val(2), 5) == [2, 3, 4]
    @test neighbors(bg, Val(2), 6) == [1, 3, 4]
    @test neighbors(bg, Val(2), 7) == [1, 2, 4]
    @test neighbors(bg, Val(2), 8) == [1, 2, 3]
end;

## Adjacency graph (fig 3.1 of "What color is your Jacobian?")

@testset "AdjacencyGraph" begin
    A = sparse([
        1 0 0 0 0 1 1 1
        0 1 0 0 1 0 1 1
        0 0 1 0 1 1 0 1
        0 0 0 1 1 1 1 0
    ])

    ag = AdjacencyGraph(transpose(A) * A)
    @test length(ag) == 8
    @test neighbors(ag, 1) == [6, 7, 8]
    @test neighbors(ag, 2) == [5, 7, 8]
    @test neighbors(ag, 3) == [5, 6, 8]
    @test neighbors(ag, 4) == [5, 6, 7]
    @test neighbors(ag, 5) == [2, 3, 4, 6, 7, 8]
    @test neighbors(ag, 6) == [1, 3, 4, 5, 7, 8]
    @test neighbors(ag, 7) == [1, 2, 4, 5, 6, 8]
    @test neighbors(ag, 8) == [1, 2, 3, 5, 6, 7]
end
