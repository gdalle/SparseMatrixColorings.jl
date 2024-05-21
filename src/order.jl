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

function vertices(ag::AdjacencyGraph, ::NaturalOrder)
    return 1:length(ag)
end

function vertices(bg::BipartiteGraph, ::Val{side}, ::NaturalOrder) where {side}
    return 1:length(bg, Val(side))
end

"""
    RandomOrder(rng=default_rng())

Order vertices with a random permutation.
"""
struct RandomOrder{R<:AbstractRNG} <: AbstractOrder
    rng::R
end

RandomOrder() = RandomOrder(default_rng())

function vertices(ag::AdjacencyGraph, order::RandomOrder)
    return randperm(order.rng, length(ag))
end

function vertices(bg::BipartiteGraph, ::Val{side}, order::RandomOrder) where {side}
    return randperm(order.rng, length(bg, Val(side)))
end

"""
    LargestFirst()

Order vertices by decreasing degree.
"""
struct LargestFirst <: AbstractOrder end

function vertices(ag::AdjacencyGraph, ::LargestFirst)
    criterion(v) = degree(ag, v)
    return sort(1:length(ag); by=criterion, rev=true)
end

function vertices(bg::BipartiteGraph, ::Val{side}, ::LargestFirst) where {side}
    criterion(v) = degree(bg, Val(side), v)
    return sort(1:length(bg, Val(side)); by=criterion, rev=true)
end
