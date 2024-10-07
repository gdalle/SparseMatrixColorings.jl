using LinearAlgebra
using SparseArrays
using SparseMatrixColorings:
    SparsityPatternCSC,
    AdjacencyGraph,
    AdjacencyFromBipartiteGraph,
    BipartiteGraph,
    degree,
    degree_dist2,
    nb_vertices,
    nb_edges,
    neighbors,
    star_coloring,
    acyclic_coloring
using Test

## SparsityPatternCSC

@testset "SparsityPatternCSC" begin
    @testset "Transpose" begin
        @test all(1:1000) do _
            A = sprand(rand(100:1000), rand(100:1000), 0.1)
            S = SparsityPatternCSC(A)
            Sᵀ = transpose(S)
            Sᵀ_true = SparsityPatternCSC(sparse(transpose(A)))
            Sᵀ.colptr == Sᵀ_true.colptr && Sᵀ.rowval == Sᵀ_true.rowval
        end
    end
    @testset "size" begin
        A = spzeros(10, 20)
        S = SparsityPatternCSC(A)
        @test size(A) == size(S)
        @test_throws BoundsError size(A, 0)
        @test size(A, 1) == size(S, 1)
        @test size(A, 2) == size(S, 2)
        @test size(A, 3) == size(S, 3)
        @test axes(A, 1) == axes(S, 1)
        @test axes(A, 2) == axes(S, 2)
    end
    @testset "getindex" begin
        A = sprand(Bool, 100, 100, 0.1)
        S = SparsityPatternCSC(A)
        @test all(zip(axes(S, 1), axes(S, 2))) do (i, j)
            A[i, j] == S[i, j]
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

@testset "AdjacencyFromBipartiteGraph" begin
    A = sparse([
        1 0 0 0 0 1 1 1
        0 1 0 0 1 0 1 1
        0 0 1 0 1 1 0 1
        0 0 0 1 1 1 1 0
    ])

    abg = AdjacencyFromBipartiteGraph(A)

    @test nb_vertices(abg) == 4 + 8
    # neighbors of columns
    @test collect(neighbors(abg, 1)) == 8 .+ [1]
    @test collect(neighbors(abg, 2)) == 8 .+ [2]
    @test collect(neighbors(abg, 3)) == 8 .+ [3]
    @test collect(neighbors(abg, 4)) == 8 .+ [4]
    @test collect(neighbors(abg, 5)) == 8 .+ [2, 3, 4]
    @test collect(neighbors(abg, 6)) == 8 .+ [1, 3, 4]
    @test collect(neighbors(abg, 7)) == 8 .+ [1, 2, 4]
    @test collect(neighbors(abg, 8)) == 8 .+ [1, 2, 3]
    # neighbors of rows
    @test collect(neighbors(abg, 8 + 1)) == [1, 6, 7, 8]
    @test collect(neighbors(abg, 8 + 2)) == [2, 5, 7, 8]
    @test collect(neighbors(abg, 8 + 3)) == [3, 5, 6, 8]
    @test collect(neighbors(abg, 8 + 4)) == [4, 5, 6, 7]

    # TODO: remove once we have better tests, this is just to check whether it runs
    @test length(star_coloring(abg, NaturalOrder())[1]) == 12
    @test length(acyclic_coloring(abg, NaturalOrder())[1]) == 12
end
