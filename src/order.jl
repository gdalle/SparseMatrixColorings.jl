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

"""
    DynamicDegreeBasedOrder{degtype,direction}

Common framework for various orders based on dynamic degrees.

# Type parameters

- `degtype::Symbol`: can be `:forward` (for the forward degree) or `:back` (for the back degree)
- `direction::Symbol`: can be `:up` (if the order is defined from lowest to highest, i.e. `1` to `n`) or `:down` (if the order is defined from highest to lowest, i.e. `n` to `1`)

# References

- [_ColPack: Software for graph coloring and related problems in scientific computing_](https://dl.acm.org/doi/10.1145/2513109.2513110), Gebremedhin et al. (2013), Section 5
"""
struct DynamicDegreeBasedOrder{degtype,direction} <: AbstractOrder end

function vertices(
    g::Graph, ::DynamicDegreeBasedOrder{degtype,direction}
) where {degtype,direction}
    # Initialize degrees
    if (degtype == :back && direction == :up) || (degtype == :forward && direction == :down)
        degrees = [0 + 1 for v in vertices(g)]  # back degree increases
    else
        degrees = [degree(g, v) + 1 for v in vertices(g)]  # forward degree decreases
    end

    # Initialize buckets and positions
    buckets = [Int[] for d in 1:(maximum_degree(g) + 1)]
    positions = zeros(Int, length(g))
    for v in vertices(g)
        d = degrees[v]
        push!(buckets[d], v)
        positions[v] = length(buckets[d])
    end

    order = Int[]

    for _ in 1:length(g)
        # Pick the candidate as the remaining vertex with largest or smallest degree
        if direction == :up
            candidate_d = findlast(!isempty, buckets)  # start with largest degree
        else
            candidate_d = findfirst(!isempty, buckets)  # start with smallest degree
        end
        candidate_bucket = buckets[candidate_d]
        candidate = pop!(candidate_bucket)
        if direction == :up
            push!(order, candidate)  # order grows from 1 to n
        else
            pushfirst!(order, candidate)  # order grows from n to 1
        end

        for v in neighbors(g, candidate)
            @assert v != candidate
            d, p = degrees[v], positions[v]
            # Discard neighbor if it has already been ordered
            d == -1 && continue

            # Modify the neighbor's old bucket by swapping and popping
            neighbor_bucket = buckets[d]
            w = neighbor_bucket[end]
            neighbor_bucket[p] = w
            neighbor_bucket[end] = v
            positions[w] = p
            pop!(neighbor_bucket)

            # Modify the neighbor's new bucket by pushing and reindexing
            if (degtype == :back && direction == :up) ||
                (degtype == :forward && direction == :down)
                d_new = d + 1  # back degree increases
            else
                d_new = d - 1  # forward degree decreases
            end
            neighbor_new_bucket = buckets[d_new]
            push!(neighbor_new_bucket, v)
            degrees[v] = d_new
            positions[v] = length(neighbor_new_bucket)
        end

        degrees[candidate] = -1
        positions[candidate] = -1
    end

    return order
end

"""
    IncidenceDegree()

Order vertices with the incidence degree heuristic.

# See also

- [`DynamicDegreeBasedOrder`](@ref)
"""
const IncidenceDegree = DynamicDegreeBasedOrder{:back,:up}

Base.show(io::IO, ::IncidenceDegree) = print(io, "IncidenceDegree()")

"""
    SmallestLast()

Order vertices with the smallest last heuristic.

# See also

- [`DynamicDegreeBasedOrder`](@ref)
"""
const SmallestLast = DynamicDegreeBasedOrder{:back,:down}

Base.show(io::IO, ::SmallestLast) = print(io, "SmallestLast()")

"""
    DynamicLargestFirst()

Order vertices with the dynamic largest first heuristic.

# See also
    
- [`DynamicDegreeBasedOrder`](@ref)
"""
const DynamicLargestFirst = DynamicDegreeBasedOrder{:forward,:up}

Base.show(io::IO, ::DynamicLargestFirst) = print(io, "DynamicLargestFirst()")
