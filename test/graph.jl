using LinearAlgebra
using SparseArrays
using SparseMatrixColorings:
    Graph, adjacency_graph, bipartite_graph, degree, nb_vertices, nb_edges, neighbors
using Test

## Standard graph

@testset "Graph" begin
    g = Graph{true}(sparse([
        1 0 1 1
        1 1 0 0
        0 0 0 1
    ]))
    gᵀ = transpose(g)

    @test nb_vertices(g) == 4
    @test nb_edges(g) == 6
    @test nnz(g) == 6
    @test neighbors(g, 1) == [1, 2]
    @test neighbors(g, 2) == [2]
    @test neighbors(g, 3) == [1]
    @test neighbors(g, 4) == [1, 3]
    @test degree(g, 1) == 2
    @test degree(g, 2) == 1
    @test degree(g, 3) == 1
    @test degree(g, 4) == 2

    @test nb_vertices(gᵀ) == 3
    @test nb_edges(gᵀ) == 6
    @test nnz(gᵀ) == 6
    @test neighbors(gᵀ, 1) == [1, 3, 4]
    @test neighbors(gᵀ, 2) == [1, 2]
    @test neighbors(gᵀ, 3) == [4]
    @test degree(gᵀ, 1) == 3
    @test degree(gᵀ, 2) == 2
    @test degree(gᵀ, 3) == 1

    g = Graph{false}(sparse([
        1 0 1
        1 1 0
        0 0 0
    ]))

    @test nb_vertices(g) == 3
    @test nb_edges(g) == 2
    @test nnz(g) == 4
    @test collect(neighbors(g, 1)) == [2]
    @test collect(neighbors(g, 2)) == Int[]
    @test collect(neighbors(g, 3)) == [1]
    @test degree(g, 1) == 1
    @test degree(g, 2) == 0
    @test degree(g, 3) == 1
end;

## Bipartite graph (fig 3.1 of "What color is your Jacobian?")

@testset "BipartiteGraph" begin
    A = sparse([
        1 0 0 0 0 1 1 1
        0 1 0 0 1 0 1 1
        0 0 1 0 1 1 0 1
        0 0 0 1 1 1 1 0
    ])

    bg = bipartite_graph(A; symmetric_pattern=false)
    @test_throws DimensionMismatch bipartite_graph(A; symmetric_pattern=true)
    @test nb_vertices(bg, Val(1)) == 4
    @test nb_vertices(bg, Val(2)) == 8
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

    A = sparse([
        1 0 1 1
        0 1 0 1
        1 0 1 0
        1 1 0 1
    ])
    bg = bipartite_graph(A; symmetric_pattern=true)
    @test nb_vertices(bg, Val(1)) == 4
    @test nb_vertices(bg, Val(2)) == 4
    # neighbors of rows and columns
    @test neighbors(bg, Val(1), 1) == neighbors(bg, Val(2), 1) == [1, 3, 4]
    @test neighbors(bg, Val(1), 2) == neighbors(bg, Val(2), 2) == [2, 4]
    @test neighbors(bg, Val(1), 3) == neighbors(bg, Val(2), 3) == [1, 3]
    @test neighbors(bg, Val(1), 4) == neighbors(bg, Val(2), 4) == [1, 2, 4]
end;

## Adjacency graph (fig 3.1 of "What color is your Jacobian?")

@testset "AdjacencyGraph" begin
    A = sparse([
        1 0 0 0 0 1 1 1
        0 1 0 0 1 0 1 1
        0 0 1 0 1 1 0 1
        0 0 0 1 1 1 1 0
    ])

    B = transpose(A) * A
    g = adjacency_graph(B - Diagonal(B))
    @test nb_vertices(g) == 8
    @test collect(neighbors(g, 1)) == [6, 7, 8]
    @test collect(neighbors(g, 2)) == [5, 7, 8]
    @test collect(neighbors(g, 3)) == [5, 6, 8]
    @test collect(neighbors(g, 4)) == [5, 6, 7]
    @test collect(neighbors(g, 5)) == [2, 3, 4, 6, 7, 8]
    @test collect(neighbors(g, 6)) == [1, 3, 4, 5, 7, 8]
    @test collect(neighbors(g, 7)) == [1, 2, 4, 5, 6, 8]
    @test collect(neighbors(g, 8)) == [1, 2, 3, 5, 6, 7]
end

@testset "Transpose" begin
    for _ in 1:1000
        A = sprand(rand(100:1000), rand(100:1000), 0.1)
        g = Graph{true}(A)
        gᵀ = transpose(g)
        gᵀ_true = Graph{true}(sparse(transpose(A)))
        @test gᵀ.colptr == gᵀ_true.colptr
        @test gᵀ.rowval == gᵀ_true.rowval
    end
end
