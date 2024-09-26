"""
    AbstractOrder

Abstract supertype for the vertex order used inside [`GreedyColoringAlgorithm`](@ref).

In this algorithm, the rows and columns of a matrix form a graph, and the vertices are colored one after the other in a greedy fashion.
Depending on how the vertices are ordered, the number of colors necessary may vary.

# Subtypes

- [`NaturalOrder`](@ref)
- [`RandomOrder`](@ref)
- [`LargestFirst`](@ref)
"""
abstract type AbstractOrder end

"""
    NaturalOrder()

Instance of [`AbstractOrder`](@ref) which sorts vertices using their index in the provided graph.
"""
struct NaturalOrder <: AbstractOrder end

function vertices(g::Graph, ::NaturalOrder)
    return vertices(g)
end

function vertices(bg::BipartiteGraph, ::Val{side}, ::NaturalOrder) where {side}
    return vertices(bg, Val(side))
end

"""
    RandomOrder(rng=default_rng())

Instance of [`AbstractOrder`](@ref) which sorts vertices using a random permutation.
"""
struct RandomOrder{R<:AbstractRNG} <: AbstractOrder
    rng::R
end

RandomOrder() = RandomOrder(default_rng())

function vertices(g::Graph, order::RandomOrder)
    return randperm(order.rng, nb_vertices(g))
end

function vertices(bg::BipartiteGraph, ::Val{side}, order::RandomOrder) where {side}
    return randperm(order.rng, nb_vertices(bg, Val(side)))
end

"""
    LargestFirst()

Instance of [`AbstractOrder`](@ref) which sorts vertices using their degree in the provided graph: the largest degree comes first.
"""
struct LargestFirst <: AbstractOrder end

function vertices(g::Graph, ::LargestFirst)
    criterion(v) = degree(g, v)
    return sort(vertices(g); by=criterion, rev=true)
end

function vertices(bg::BipartiteGraph, ::Val{side}, ::LargestFirst) where {side}
    other_side = 3 - side
    n = nb_vertices(bg, Val(side))
    visited = falses(n)  # necessary for distance-2 neighborhoods
    degrees_dist2 = zeros(Int, n)
    for v in vertices(bg, Val(side))
        fill!(visited, false)
        for u in neighbors(bg, Val(side), v)
            for w in neighbors(bg, Val(other_side), u)
                if w != v && !visited[w]
                    degrees_dist2[v] += 1
                    visited[w] = true  # avoid double counting
                end
            end
        end
    end
    criterion(v) = degrees_dist2[v]
    return sort(vertices(bg, Val(side)); by=criterion, rev=true)
end
