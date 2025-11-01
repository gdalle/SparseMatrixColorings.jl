using LinearAlgebra
using SparseArrays
using SparseMatrixColorings:
    SparsityPatternCSC,
    AdjacencyGraph,
    BipartiteGraph,
    bidirectional_pattern,
    degree,
    degree_dist2,
    nb_vertices,
    nb_edges,
    neighbors
using Test

## SparsityPatternCSC

@testset "SparsityPatternCSC" begin
    @test eltype(SparsityPatternCSC(sprand(10, 10, 0.1))) == Int
    @testset "Transpose" begin
        for _ in 1:1000
            m, n = rand(100:1000), rand(100:1000)
            p = 0.05 * rand()
            A = sprand(m, n, p)
            S = SparsityPatternCSC(A)
            Sᵀ = transpose(S)
            Sᵀ_true = SparsityPatternCSC(sparse(transpose(A)))
            @test Sᵀ.colptr == Sᵀ_true.colptr
            @test Sᵀ.rowval == Sᵀ_true.rowval
        end
    end
    @testset "Bidirectional" begin
        @testset "symmetric_pattern = false" begin
            for _ in 1:1000
                m, n = rand(100:1000), rand(100:1000)
                p = 0.05 * rand()
                A = sprand(Bool, m, n, p)
                A_and_Aᵀ = [spzeros(Bool, n, n) transpose(A); A spzeros(Bool, m, m)]
                S_and_Sᵀ, edge_to_index = bidirectional_pattern(A; symmetric_pattern=false)
                @test S_and_Sᵀ.colptr == A_and_Aᵀ.colptr
                @test S_and_Sᵀ.rowval == A_and_Aᵀ.rowval
                M = SparseMatrixCSC(
                    m + n, m + n, S_and_Sᵀ.colptr, S_and_Sᵀ.rowval, edge_to_index
                )
                @test issymmetric(M)
            end
        end
        @testset "symmetric_pattern = true" begin
            for _ in 1:1000
                m = rand(100:1000)
                p = 0.05 * rand()
                A = sparse(Symmetric(sprand(Bool, m, m, p)))
                A_and_Aᵀ = [spzeros(Bool, m, m) transpose(A); A spzeros(Bool, m, m)]
                S_and_Sᵀ, edge_to_index = bidirectional_pattern(A; symmetric_pattern=true)
                @test S_and_Sᵀ.colptr == A_and_Aᵀ.colptr
                @test S_and_Sᵀ.rowval == A_and_Aᵀ.rowval
                M = SparseMatrixCSC(
                    2 * m, 2 * m, S_and_Sᵀ.colptr, S_and_Sᵀ.rowval, edge_to_index
                )
                @test issymmetric(M)
            end
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
    @test eltype(bg) == Int
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

    g = AdjacencyGraph(transpose(A) * A)
    @test eltype(g) == Int
    @test nb_vertices(g) == 8
    # wrong neighbors, it's okay they are filtered after
    @test collect(neighbors(g, 1)) == sort(vcat(1, [6, 7, 8]))
    @test collect(neighbors(g, 2)) == sort(vcat(2, [5, 7, 8]))
    @test collect(neighbors(g, 3)) == sort(vcat(3, [5, 6, 8]))
    @test collect(neighbors(g, 4)) == sort(vcat(4, [5, 6, 7]))
    @test collect(neighbors(g, 5)) == sort(vcat(5, [2, 3, 4, 6, 7, 8]))
    @test collect(neighbors(g, 6)) == sort(vcat(6, [1, 3, 4, 5, 7, 8]))
    @test collect(neighbors(g, 7)) == sort(vcat(7, [1, 2, 4, 5, 6, 8]))
    @test collect(neighbors(g, 8)) == sort(vcat(8, [1, 2, 3, 5, 6, 7]))
    # right degree
    @test degree(g, 1) == 3
    @test degree(g, 2) == 3
    @test degree(g, 3) == 3
    @test degree(g, 4) == 3
    @test degree(g, 5) == 6
    @test degree(g, 6) == 6
    @test degree(g, 7) == 6
    @test degree(g, 8) == 6

    g = AdjacencyGraph(transpose(A) * A; augmented_graph=true)
    # wrong degree
    @test degree(g, 1) == 4
    @test degree(g, 2) == 4
    @test degree(g, 3) == 4
    @test degree(g, 4) == 4
    @test degree(g, 5) == 7
    @test degree(g, 6) == 7
    @test degree(g, 7) == 7
    @test degree(g, 8) == 7
end
