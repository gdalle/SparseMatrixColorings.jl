## Forest

"""
$TYPEDEF

Structure that provides fast union-find operations for constructing a forest during acyclic coloring and bicoloring.

# Fields

$TYPEDFIELDS
"""
mutable struct Forest{T<:Integer}
    "current number of edges in the forest"
    num_edges::T
    "current number of distinct trees in the forest"
    num_trees::T
    "dictionary mapping each edge represented as a tuple of vertices to its unique integer index"
    intmap::Dict{Tuple{T,T},T}
    "vector storing the index of a parent in the tree for each edge, used in union-find operations"
    parents::Vector{T}
    "vector approximating the depth of each tree to optimize path compression"
    ranks::Vector{T}
end

function Forest{T}(n::Integer) where {T<:Integer}
    num_edges = zero(T)
    num_trees = zero(T)
    intmap = Dict{Tuple{T,T},T}()
    sizehint!(intmap, n)
    parents = collect(Base.OneTo(T(n)))
    ranks = zeros(T, T(n))
    return Forest{T}(num_edges, num_trees, intmap, parents, ranks)
end

function Base.push!(forest::Forest{T}, edge::Tuple{T,T}) where {T<:Integer}
    forest.num_edges += 1
    forest.intmap[edge] = forest.num_edges
    forest.num_trees += one(T)
    return forest
end

function _find_root!(parents::Vector{T}, index_edge::T) where {T<:Integer}
    p = parents[index_edge]
    if parents[p] != p
        parents[index_edge] = p = _find_root!(parents, p)
    end
    return p
end

function find_root!(forest::Forest{T}, edge::Tuple{T,T}) where {T<:Integer}
    return _find_root!(forest.parents, forest.intmap[edge])
end

function root_union!(forest::Forest{T}, index_edge1::T, index_edge2::T) where {T<:Integer}
    parents = forest.parents
    rks = forest.ranks
    rank1 = rks[index_edge1]
    rank2 = rks[index_edge2]

    if rank1 < rank2
        index_edge1, index_edge2 = index_edge2, index_edge1
    elseif rank1 == rank2
        rks[index_edge1] += one(T)
    end
    parents[index_edge2] = index_edge1
    forest.num_trees -= one(T)
    return nothing
end
