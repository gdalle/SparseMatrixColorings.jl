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
- [`PerfectEliminationOrder`](@ref)
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

function vertices(bg::BipartiteGraph{T}, ::Val{side}, ::LargestFirst) where {T,side}
    other_side = 3 - side
    n = nb_vertices(bg, Val(side))
    visited = falses(n)  # necessary for distance-2 neighborhoods
    degrees_dist2 = zeros(T, n)
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

struct DegreeBuckets{T}
    degrees::Vector{T}
    bucket_storage::Vector{T}
    bucket_low::Vector{T}
    bucket_high::Vector{T}
    positions::Vector{T}
end

function DegreeBuckets(::Type{T}, degrees::Vector{<:Integer}, dmax::Integer) where {T}
    # number of vertices per degree class
    deg_count = zeros(T, dmax + 1)
    for d in degrees
        deg_count[d + 1] += 1
    end
    # bucket limits
    bucket_high = cumsum(deg_count)
    bucket_low = vcat(zero(T), @view(bucket_high[1:(end - 1)]))
    bucket_low .+= 1
    # assign each vertex to the correct position inside its degree class
    bucket_storage = similar(degrees, T)
    positions = similar(degrees, T)
    for v in eachindex(positions, degrees)
        d = degrees[v]
        positions[v] = bucket_high[d + 1] - deg_count[d + 1] + 1
        bucket_storage[positions[v]] = v
        deg_count[d + 1] -= 1
    end
    return DegreeBuckets(degrees, bucket_storage, bucket_low, bucket_high, positions)
end

maxdeg(db::DegreeBuckets) = length(db.bucket_low) - 1

function degree_increasing(; degtype, direction)
    increasing =
        (degtype == :back && direction == :low2high) ||
        (degtype == :forward && direction == :high2low)
    return increasing
end

function mark_ordered!(db::DegreeBuckets{T}, v::Integer) where {T}
    db.degrees[v] = -1
    db.positions[v] = typemin(T)
    return nothing
end

already_ordered(db::DegreeBuckets, v::Integer) = db.degrees[v] == -1

function pop_next_candidate!(db::DegreeBuckets; direction::Symbol)
    (; bucket_storage, bucket_low, bucket_high) = db
    dmax = maxdeg(db)
    if direction == :low2high
        candidate_degree = dmax + 1
        for d in dmax:-1:0
            if bucket_high[d + 1] >= bucket_low[d + 1]  # not empty
                candidate_degree = d
                break
            end
        end
    else
        candidate_degree = -1
        for d in 0:dmax
            if bucket_high[d + 1] >= bucket_low[d + 1]  # not empty
                candidate_degree = d
                break
            end
        end
    end
    high = bucket_high[candidate_degree + 1]
    candidate = bucket_storage[high]
    bucket_storage[high] = -1
    bucket_high[candidate_degree + 1] -= 1
    mark_ordered!(db, candidate)
    return candidate
end

function update_bucket!(db::DegreeBuckets, v::Integer; degtype, direction)
    (; degrees, bucket_storage, bucket_low, bucket_high, positions) = db
    d, p = degrees[v], positions[v]
    low, high = bucket_low[d + 1], bucket_high[d + 1]
    # select previous or next bucket for the move
    if degree_increasing(; degtype, direction)
        # put v at the end of its bucket by swapping
        w = bucket_storage[high]
        bucket_storage[p] = w
        bucket_storage[high] = v
        positions[w] = p
        positions[v] = high
        # move v to the beginning of the next bucket (mind the gap)
        d_new = d + 1
        low_new, high_new = bucket_low[d_new + 1], bucket_high[d_new + 1]
        bucket_storage[low_new - 1] = v
        # update v stats
        degrees[v] = d_new
        positions[v] = low_new - 1
        # grow next bucket to the left, shrink current one from the right
        bucket_low[d_new + 1] -= 1
        bucket_high[d + 1] -= 1
    else
        # put v at the beginning of its bucket by swapping
        w = bucket_storage[low]
        bucket_storage[p] = w
        bucket_storage[low] = v
        positions[w] = p
        positions[v] = low
        # move v to the end of the previous bucket (mind the gap)
        d_new = d - 1
        low_new, high_new = bucket_low[d_new + 1], bucket_high[d_new + 1]
        bucket_storage[high_new + 1] = v
        # update v stats
        degrees[v] = d_new
        positions[v] = high_new + 1
        # grow previous bucket to the right, shrink current one from the left
        bucket_high[d_new + 1] += 1
        bucket_low[d + 1] += 1
    end
    return nothing
end

function vertices(
    g::AdjacencyGraph{T}, ::DynamicDegreeBasedOrder{degtype,direction}
) where {T<:Integer,degtype,direction}
    if degree_increasing(; degtype, direction)
        degrees = zeros(T, nb_vertices(g))
    else
        degrees = [degree(g, v) for v in vertices(g)]
    end
    db = DegreeBuckets(T, degrees, maximum_degree(g))
    π = T[]
    sizehint!(π, nb_vertices(g))
    for _ in 1:nb_vertices(g)
        u = pop_next_candidate!(db; direction)
        direction == :low2high ? push!(π, u) : pushfirst!(π, u)
        for v in neighbors(g, u)
            !has_diagonal(g) || (u == v && continue)
            already_ordered(db, v) && continue
            update_bucket!(db, v; degtype, direction)
        end
    end
    return π
end

function vertices(
    g::BipartiteGraph{T}, ::Val{side}, ::DynamicDegreeBasedOrder{degtype,direction}
) where {T<:Integer,side,degtype,direction}
    other_side = 3 - side
    # compute dist-2 degrees in an optimized way
    n = nb_vertices(g, Val(side))
    degrees_dist2 = zeros(T, n)
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
        degrees = zeros(T, n)
    else
        degrees = degrees_dist2
    end
    maxd2 = maximum(degrees_dist2)
    db = DegreeBuckets(T, degrees, maxd2)
    π = T[]
    sizehint!(π, n)
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
    PerfectEliminationOrder(elimination_algorithm=CliqueTrees.MCS())

Instance of [`AbstractOrder`](@ref) which computes a perfect elimination ordering when the underlying graph is [chordal](https://en.wikipedia.org/wiki/Chordal_graph). For non-chordal graphs, it computes a suboptimal ordering.

The `elimination_algorithm` must be an instance of `CliqueTrees.EliminationAlgorithm`.

!!! warning
    This order can only be applied for symmetric or bidirectional coloring problems. Furthermore, its theoretical guarantees only hold for decompression by substitution.

!!! danger
    This order is implemented as a package extension and requires loading [CliqueTrees.jl](https://github.com/AlgebraicJulia/CliqueTrees.jl).

# References

- [Simple Linear-Time Algorithms to Test Chordality of Graphs, Test Acyclicity of Hypergraphs, and Selectively Reduce Acyclic Hypergraphs](https://epubs.siam.org/doi/10.1137/0213035), Tarjan and Yannakakis (1984)
"""
struct PerfectEliminationOrder{E} <: AbstractOrder
    elimination_algorithm::E
end
