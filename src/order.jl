"""
    AbstractOrder

Abstract supertype for the vertex order used inside [`GreedyColoringAlgorithm`](@ref).

In this algorithm, the rows and columns of a matrix form a graph, and the vertices are colored one after the other in a greedy fashion.
Depending on how the vertices are ordered, the number of colors necessary may vary.

# Options

- [`NaturalOrder`](@ref)
- [`RandomOrder`](@ref)
- [`LargestFirst`](@ref)
- [`IncidenceDegree`](@ref) (experimental)
- [`SmallestLast`](@ref) (experimental)
- [`DynamicLargestFirst`](@ref) (experimental)
"""
abstract type AbstractOrder end

"""
    NaturalOrder()

Instance of [`AbstractOrder`](@ref) which sorts vertices using their index in the provided graph.
"""
struct NaturalOrder <: AbstractOrder end

function vertices(g::AdjacencyGraph, ::NaturalOrder)
    return vertices(g)
end

function vertices(bg::BipartiteGraph, ::Val{side}, ::NaturalOrder) where {side}
    return vertices(bg, Val(side))
end

"""
    RandomOrder(rng=default_rng(), seed=nothing)

Instance of [`AbstractOrder`](@ref) which sorts vertices using a random permutation, generated from `rng` with a given `seed`.

- If `seed = nothing`, the `rng` will never be re-seeded. Therefore, two consecutive calls to `vertices(g, order)` will give different results.
- Otherwise, the `rng` will be re-seeded before each sample. Therefore, two consecutive calls to `vertices(g, order)` will give the same result.

!!! warning
    Do not use a seed with the `default_rng()`, otherwise you will affect the global state of your program.
    If you need reproducibility, create a new `rng` specifically for your `RandomOrder`.
    The package [StableRNGs.jl](https://github.com/JuliaRandom/StableRNGs.jl) offers random number generators whose behavior is stable across Julia versions.
"""
struct RandomOrder{R<:AbstractRNG,S} <: AbstractOrder
    rng::R
    seed::S
end

RandomOrder(rng::AbstractRNG) = RandomOrder(rng, nothing)
RandomOrder() = RandomOrder(default_rng())

function vertices(g::AdjacencyGraph, order::RandomOrder)
    (; rng, seed) = order
    if isnothing(seed)
        return randperm(rng, nb_vertices(g))
    else
        return randperm(Random.seed!(rng, seed), nb_vertices(g))
    end
end

function vertices(bg::BipartiteGraph, ::Val{side}, order::RandomOrder) where {side}
    (; rng, seed) = order
    if isnothing(seed)
        return randperm(rng, nb_vertices(bg, Val(side)))
    else
        return randperm(Random.seed!(rng, seed), nb_vertices(bg, Val(side)))
    end
end

"""
    LargestFirst()

Instance of [`AbstractOrder`](@ref) which sorts vertices using their degree in the provided graph: the largest degree comes first.
"""
struct LargestFirst <: AbstractOrder end

function vertices(g::AdjacencyGraph, ::LargestFirst)
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

"""
    DynamicDegreeBasedOrder{degtype,direction}

Instance of [`AbstractOrder`](@ref) which sorts vertices using a dynamically computed degree.

!!! danger
    This order is still experimental and needs more tests, correctness is not yet guaranteed.

# Type parameters

- `degtype::Symbol`: can be `:forward` (for the forward degree) or `:back` (for the back degree)
- `direction::Symbol`: can be `:low2high` (if the order is defined from lowest to highest, i.e. `1` to `n`) or `:high2low` (if the order is defined from highest to lowest, i.e. `n` to `1`)

# Concrete variants

- [`IncidenceDegree`](@ref)
- [`SmallestLast`](@ref)
- [`DynamicLargestFirst`](@ref)

# References

- [_ColPack: Software for graph coloring and related problems in scientific computing_](https://dl.acm.org/doi/10.1145/2513109.2513110), Gebremedhin et al. (2013), Section 5
"""
struct DynamicDegreeBasedOrder{degtype,direction} <: AbstractOrder end

struct DegreeBuckets{B}
    degrees::Vector{Int}
    buckets::B
    positions::Vector{Int}
end

function DegreeBuckets(degrees::Vector{Int}, dmax)
    buckets = Dict(d => Int[] for d in 0:dmax)
    positions = similar(degrees, Int)
    for v in eachindex(degrees, positions)
        d = degrees[v]
        push!(buckets[d], v)  # TODO: optimize
        positions[v] = length(buckets[d])
    end
    return DegreeBuckets(degrees, buckets, positions)
end

function degree_increasing(; degtype, direction)
    increasing =
        (degtype == :back && direction == :low2high) ||
        (degtype == :forward && direction == :high2low)
    return increasing
end

function mark_ordered!(db::DegreeBuckets, v::Integer)
    db.degrees[v] = -1
    db.positions[v] = -1
    return nothing
end

already_ordered(db::DegreeBuckets, v::Integer) = db.degrees[v] == -1

function pop_next_candidate!(db::DegreeBuckets; direction::Symbol)
    (; buckets) = db
    if direction == :low2high
        candidate_degree = maximum(d for (d, bucket) in pairs(buckets) if !isempty(bucket))
    else
        candidate_degree = minimum(d for (d, bucket) in pairs(buckets) if !isempty(bucket))
    end
    candidate_bucket = buckets[candidate_degree]
    candidate = pop!(candidate_bucket)
    mark_ordered!(db, candidate)
    return candidate
end

function update_bucket!(db::DegreeBuckets, v::Integer; degtype, direction)
    (; degrees, buckets, positions) = db
    d, p = degrees[v], positions[v]
    bucket = buckets[d]
    # select previous or next bucket for the move
    d_new = degree_increasing(; degtype, direction) ? d + 1 : d - 1
    bucket_new = buckets[d_new]
    # put v at the end of its bucket by swapping
    w = bucket[end]
    bucket[p] = w
    positions[w] = p
    bucket[end] = v
    positions[v] = length(bucket)
    # move v from the old bucket to the new one
    @assert pop!(bucket) == v
    push!(bucket_new, v)
    degrees[v] = d_new
    positions[v] = length(bucket_new)
    return nothing
end

function vertices(
    g::AdjacencyGraph, ::DynamicDegreeBasedOrder{degtype,direction}
) where {degtype,direction}
    if degree_increasing(; degtype, direction)
        degrees = zeros(Int, nb_vertices(g))
    else
        degrees = [degree(g, v) for v in vertices(g)]
    end
    db = DegreeBuckets(degrees, maximum_degree(g))
    π = Int[]
    for _ in 1:nb_vertices(g)
        u = pop_next_candidate!(db; direction)
        direction == :low2high ? push!(π, u) : pushfirst!(π, u)
        for v in neighbors(g, u)
            already_ordered(db, v) && continue
            update_bucket!(db, v; degtype, direction)
        end
    end
    return π
end

function vertices(
    g::BipartiteGraph, ::Val{side}, ::DynamicDegreeBasedOrder{degtype,direction}
) where {side,degtype,direction}
    other_side = 3 - side
    # compute dist-2 degrees in an optimized way
    n = nb_vertices(g, Val(side))
    degrees_dist2 = zeros(Int, n)
    dist2_neighbor = falses(n)
    for v in vertices(g, Val(side))
        fill!(dist2_neighbor, false)
        for w1 in neighbors(g, Val(side), v)
            for w2 in neighbors(g, Val(other_side), w1)
                dist2_neighbor[w2] = true
            end
        end
        degrees_dist2[v] = sum(dist2_neighbor)
    end
    if degree_increasing(; degtype, direction)
        degrees = zeros(Int, n)
    else
        degrees = degrees_dist2
    end
    maxd2 = maximum(degrees_dist2)
    db = DegreeBuckets(degrees, maxd2)
    π = Int[]
    visited = falses(n)
    for _ in 1:nb_vertices(g, Val(side))
        u = pop_next_candidate!(db; direction)
        direction == :low2high ? push!(π, u) : pushfirst!(π, u)
        for w in neighbors(g, Val(side), u)
            for v in neighbors(g, Val(other_side), w)
                if v == u || visited[v]
                    continue
                else
                    visited[v] = true
                end
                already_ordered(db, v) && continue
                update_bucket!(db, v; degtype, direction)
            end
        end
        fill!(visited, false)
    end
    return π
end

"""
    IncidenceDegree()

Instance of [`AbstractOrder`](@ref) which sorts vertices from lowest to highest using the dynamic back degree.

!!! danger
    This order is still experimental and needs more tests, correctness is not yet guaranteed.

# See also

- [`DynamicDegreeBasedOrder`](@ref)
"""
const IncidenceDegree = DynamicDegreeBasedOrder{:back,:low2high}

"""
    SmallestLast()

Instance of [`AbstractOrder`](@ref) which sorts vertices from highest to lowest using the dynamic back degree.

!!! danger
    This order is still experimental and needs more tests, correctness is not yet guaranteed.

# See also

- [`DynamicDegreeBasedOrder`](@ref)
"""
const SmallestLast = DynamicDegreeBasedOrder{:back,:high2low}

"""
    DynamicLargestFirst()

Instance of [`AbstractOrder`](@ref) which sorts vertices from lowest to highest using the dynamic forward degree.

!!! danger
    This order is still experimental and needs more tests, correctness is not yet guaranteed.

# See also
    
- [`DynamicDegreeBasedOrder`](@ref)
"""
const DynamicLargestFirst = DynamicDegreeBasedOrder{:forward,:low2high}

"""
    PerfectEliminationOrder

Instance of [`AbstractOrder`](@ref) which computes a perfect elimination ordering when the underlying graph is [chordal](https://en.wikipedia.org/wiki/Chordal_graph). For non-chordal graphs, it computes a suboptimal ordering.

!!! warning
    This order can only be applied for symmetric or bidirectional coloring problems.

!!! danger
    This order is implemented as a package extension and requires loading [CliqueTrees.jl](https://github.com/AlgebraicJulia/CliqueTrees.jl).

# References

- [Simple Linear-Time Algorithms to Test Chordality of Graphs, Test Acyclicity of Hypergraphs, and Selectively Reduce Acyclic Hypergraphs](https://epubs.siam.org/doi/10.1137/0213035), Tarjan and Yannakakis (1984)
"""
struct PerfectEliminationOrder <: AbstractOrder end
