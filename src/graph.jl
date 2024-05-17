"""
    Graph{Ti<:Integer}

Graph structure on vertices of type `Ti`.
"""
struct Graph{Ti<:Integer}
    colptr::Vector{Ti}
    rowval::Vector{Ti}
end

Graph(A::SparseMatrixCSC) = Graph(A.colptr, A.rowval)

Base.length(g::Graph) = length(g.colptr) - 1

"""
    neighbors(g::Graph, v::Integer)

Return the neighbors of vertex `v` in graph `g` as an iterable.
"""
neighbors(g::Graph, v::Integer) = @view g.rowval[g.colptr[v]:(g.colptr[v + 1] - 1)]

## Bipartite graph

function bipartite_graph(J::SparseMatrixCSC{Tv}) where {Tv}
    m, n = size(J)
    Jt = transpose(J)
    O_mm = spzeros(Tv, m, m)
    O_nn = spzeros(Tv, n, n)
    A = [
        O_mm J
        Jt O_nn
    ]
    return Graph(A)
end

## Adjacency graph

function adjacency_graph(H::SparseMatrixCSC)
    return Graph(H - Diagonal(H))
end
