"""
    AbstractOrder

Abstract supertype for vertex ordering schemes.

# Subtypes

- [`NaturalOrder`](@ref)
- [`RandomOrder`](@ref)
- [`LargestFirst`](@ref)
"""
abstract type AbstractOrder end

"""
    NaturalOrder()

Order vertices as they come in the graph.
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

Order vertices with a random permutation.
"""
struct RandomOrder{R<:AbstractRNG} <: AbstractOrder
    rng::R
end

RandomOrder() = RandomOrder(default_rng())

function vertices(g::Graph, order::RandomOrder)
    return randperm(order.rng, length(g))
end

function vertices(bg::BipartiteGraph, ::Val{side}, order::RandomOrder) where {side}
    return randperm(order.rng, length(bg, Val(side)))
end

"""
    LargestFirst()

Order vertices by decreasing degree.
"""
struct LargestFirst <: AbstractOrder end

function vertices(g::Graph, ::LargestFirst)
    criterion(v) = degree(g, v)
    return sort(vertices(g); by=criterion, rev=true)
end

function vertices(bg::BipartiteGraph, ::Val{side}, ::LargestFirst) where {side}
    criterion(v) = degree(bg, Val(side), v)
    return sort(vertices(bg, Val(side)); by=criterion, rev=true)
end
