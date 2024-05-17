using SparseArrays
using SparseMatrixColorings: Graph, adjacency_graph, bipartite_graph, neighbors
using Test

## Standard graph

g = Graph(sparse([
    1 0 1
    1 1 0
    0 0 0
]))

@test length(g) == 3
@test neighbors(g, 1) == [1, 2]
@test neighbors(g, 2) == [2]
@test neighbors(g, 3) == [1]

## Bipartite graph (fig 3.1)

J = sparse([
    1 0 0 0 0 1 1 1
    0 1 0 0 1 0 1 1
    0 0 1 0 1 1 0 1
    0 0 0 1 1 1 1 0
])

bg = bipartite_graph(J)
@test length(bg) == 4 + 8
# neighbors of rows
@test neighbors(bg, 1) == 4 .+ [1, 6, 7, 8]
@test neighbors(bg, 2) == 4 .+ [2, 5, 7, 8]
@test neighbors(bg, 3) == 4 .+ [3, 5, 6, 8]
@test neighbors(bg, 4) == 4 .+ [4, 5, 6, 7]
# neighbors of columns
@test neighbors(bg, 4 + 1) == [1]
@test neighbors(bg, 4 + 2) == [2]
@test neighbors(bg, 4 + 3) == [3]
@test neighbors(bg, 4 + 4) == [4]
@test neighbors(bg, 4 + 5) == [2, 3, 4]
@test neighbors(bg, 4 + 6) == [1, 3, 4]
@test neighbors(bg, 4 + 7) == [1, 2, 4]
@test neighbors(bg, 4 + 8) == [1, 2, 3]
