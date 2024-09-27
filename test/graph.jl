using LinearAlgebra
using SparseArrays
using SparseMatrixColorings:
    SparsityPatternCSC,
    AdjacencyGraph,
    BipartiteGraph,
    degree,
    degree_dist2,
    nb_vertices,
    nb_edges,
    neighbors
using Test

## SparsityPatternCSC

@testset "SparsityPatternCSC" begin
    @testset "Transpose" begin
        for _ in 1:1000
            A = sprand(rand(100:1000), rand(100:1000), 0.1)
            S = SparsityPatternCSC(A)
            Sᵀ = transpose(S)
            Sᵀ_true = SparsityPatternCSC(sparse(transpose(A)))
            @test Sᵀ.colptr == Sᵀ_true.colptr
            @test Sᵀ.rowval == Sᵀ_true.rowval
        end
    end
end

## Bipartite graph (fig 3.1 of "What color is your Jacobian?")

@testset "BipartiteGraph" begin
    A = sparse([
        1 0 0 0 0 1 1 1
        0 1 0 0 1 0 1 1
        0 0 1 0 1 1 0 1
        0 0 0 1 1 1 1 0
    ])

    bg = BipartiteGraph(A; symmetric_pattern=false)
    @test_throws DimensionMismatch BipartiteGraph(A; symmetric_pattern=true)
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
    @test degree_dist2(bg, Val(2), 1) == 3
    @test degree_dist2(bg, Val(2), 2) == 3
    @test degree_dist2(bg, Val(2), 3) == 3
    @test degree_dist2(bg, Val(2), 4) == 3
    @test degree_dist2(bg, Val(2), 5) == 6
    @test degree_dist2(bg, Val(2), 6) == 6
    @test degree_dist2(bg, Val(2), 7) == 6
    @test degree_dist2(bg, Val(2), 8) == 6

    A = sparse([
        1 0 1 1
        0 1 0 1
        1 0 1 0
        1 1 0 1
    ])
    bg = BipartiteGraph(A; symmetric_pattern=true)
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
    g = AdjacencyGraph(B - Diagonal(B))
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
