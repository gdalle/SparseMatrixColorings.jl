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
    degrees = map(Base.Fix1(degree, g), vertices(g))
    criterion(v) = degrees[v]
    return sort(vertices(g); by=criterion, rev=true)
end

function vertices(bg::BipartiteGraph{T}, ::Val{side}, ::LargestFirst) where {T,side}
    other_side = 3 - side
    n = nb_vertices(bg, Val(side))
    visited = zeros(T, n)  # necessary for distance-2 neighborhoods
    degrees_dist2 = zeros(T, n)
    for v in vertices(bg, Val(side))
        for u in neighbors(bg, Val(side), v)
            for w in neighbors(bg, Val(other_side), u)
                if w != v && visited[w] != v
                    degrees_dist2[v] += 1
                    visited[w] = v  # avoid double counting
                end
            end
        end
    end
    # Recycle the vector visited to store the ordering
    visited .= 1:n
    criterion(v) = degrees_dist2[v]
    return sort!(visited; by=criterion, rev=true)
end

"""
    DynamicDegreeBasedOrder{degtype,direction}(; reproduce_colpack=false)

Instance of [`AbstractOrder`](@ref) which sorts vertices using a dynamically computed degree.

This order works by assigning vertices to buckets based on their dynamic degree, and then updating buckets iteratively by transfering vertices between them.

# Type parameters

- `degtype::Symbol`: can be `:forward` (for the forward degree) or `:back` (for the back degree)
- `direction::Symbol`: can be `:low2high` (if the order is defined from lowest to highest, i.e. `1` to `n`) or `:high2low` (if the order is defined from highest to lowest, i.e. `n` to `1`)

# Concrete variants

- [`IncidenceDegree`](@ref)
- [`SmallestLast`](@ref)
- [`DynamicLargestFirst`](@ref)

# Settings

- `reproduce_colpack::Bool`: whether to manage the buckets in the exact same way as the original ColPack implementation.
  - When `reproduce_colpack=true`, we always append and remove vertices at the end of a bucket (unilateral).
  - When `reproduce_colpack=false` (the default), we can append and remove vertices either at the start or at the end of a bucket (bilateral).

Allowing modifications on both sides of a bucket enables storage optimization, with a single fixed-size vector for all buckets instead of one dynamically-sized vector per bucket.
As a result, the default setting `reproduce_colpack=false` is slightly more memory-efficient.

# References

- [_ColPack: Software for graph coloring and related problems in scientific computing_](https://dl.acm.org/doi/10.1145/2513109.2513110), Gebremedhin et al. (2013), Section 5
"""
struct DynamicDegreeBasedOrder{degtype,direction,reproduce_colpack} <: AbstractOrder end

function DynamicDegreeBasedOrder{degtype,direction}(;
    reproduce_colpack::Bool=false
) where {degtype,direction}
    return DynamicDegreeBasedOrder{degtype,direction,reproduce_colpack}()
end

abstract type AbstractDegreeBuckets{T} end

struct DegreeBucketsColPack{T} <: AbstractDegreeBuckets{T}
    degrees::Vector{T}
    buckets::Vector{Vector{T}}
    positions::Vector{T}
end

struct DegreeBucketsSMC{T} <: AbstractDegreeBuckets{T}
    degrees::Vector{T}
    bucket_storage::Vector{T}
    bucket_low::Vector{T}
    bucket_high::Vector{T}
    positions::Vector{T}
end

function DegreeBucketsColPack(::Type{T}, degrees::Vector{T}, dmax::Integer) where {T}
    # number of vertices per degree class
    deg_count = zeros(T, dmax + 1)
    for d in degrees
        deg_count[d + 1] += 1
    end
    # one vector per bucket
    buckets = [Vector{T}(undef, deg_count[d + 1]) for d in 0:dmax]
    positions = similar(degrees, T)
    # assign each vertex to the correct local position inside its bucket
    for v in eachindex(positions, degrees)
        d = degrees[v]
        positions[v] = length(buckets[d + 1]) - deg_count[d + 1] + 1
        buckets[d + 1][positions[v]] = v
        deg_count[d + 1] -= 1
    end
    return DegreeBucketsColPack(degrees, buckets, positions)
end

function DegreeBucketsSMC(::Type{T}, degrees::Vector{T}, dmax::Integer) where {T}
    # number of vertices per degree class
    deg_count = zeros(T, dmax + 1)
    for d in degrees
        deg_count[d + 1] += 1
    end
    # bucket limits
    bucket_high = accumulate(+, deg_count)
    bucket_low = similar(bucket_high)
    bucket_low[1] = 1
    bucket_low[2:end] .= @view(bucket_high[1:(end - 1)]) .+ 1
    # assign each vertex to the correct global position inside its bucket
    bucket_storage = similar(degrees, T)
    positions = similar(degrees, T)
    for v in eachindex(positions, degrees)
        d = degrees[v]
        positions[v] = bucket_high[d + 1] - deg_count[d + 1] + 1
        bucket_storage[positions[v]] = v
        deg_count[d + 1] -= 1
    end
    return DegreeBucketsSMC(degrees, bucket_storage, bucket_low, bucket_high, positions)
end

maxdeg(db::DegreeBucketsColPack) = length(db.buckets) - 1
maxdeg(db::DegreeBucketsSMC) = length(db.bucket_low) - 1

function nonempty_bucket(db::DegreeBucketsSMC, d::Integer)
    return db.bucket_high[d + 1] >= db.bucket_low[d + 1]
end
function nonempty_bucket(db::DegreeBucketsColPack, d::Integer)
    return !isempty(db.buckets[d + 1])
end

function degree_increasing(; degtype, direction)
    increasing =
        (degtype == :back && direction == :low2high) ||
        (degtype == :forward && direction == :high2low)
    return increasing
end

function pop_next_candidate!(db::AbstractDegreeBuckets; degree_range::OrdinalRange)
    (; degrees) = db
    # degree_range is used to avoid going through the empty parts of 0:dmax
    candidate_degree = -1
    for d in degree_range
        if nonempty_bucket(db, d)
            candidate_degree = d
            break
        end
    end
    if db isa DegreeBucketsColPack
        (; buckets) = db
        bucket = buckets[candidate_degree + 1]
        candidate = pop!(bucket)
    else
        (; bucket_storage, bucket_high) = db
        high = bucket_high[candidate_degree + 1]
        candidate = bucket_storage[high]
        bucket_high[candidate_degree + 1] -= 1
    end
    # mark as ordered
    degrees[candidate] = -1
    # returning candidate degree is useful to update degree_range
    return candidate, candidate_degree
end

function update_bucket!(
    db::DegreeBucketsSMC, v::Integer, d::Integer; degtype::Symbol, direction::Symbol
)
    (; degrees, bucket_storage, bucket_low, bucket_high, positions) = db
    p = positions[v]
    # select previous or next bucket for the move
    if degree_increasing(; degtype, direction)
        high = bucket_high[d + 1]
        # move the vertex w located at the end of the current bucket to v's position
        w = bucket_storage[high]
        bucket_storage[p] = w
        positions[w] = p
        # shrink current bucket from the right
        # morally we put v at the end and then ignore it
        bucket_high[d + 1] -= 1
        # move v to the beginning of the next bucket (!= ColPack)
        d_new = d + 1
        low_new = bucket_low[d_new + 1]
        bucket_storage[low_new - 1] = v
        # grow next bucket to the left
        bucket_low[d_new + 1] -= 1
        # update v's stats
        degrees[v] = d_new
        positions[v] = low_new - 1
    else
        low = bucket_low[d + 1]
        # move the vertex w located at the start of the current bucket to v's position (!= ColPack)
        w = bucket_storage[low]
        bucket_storage[p] = w
        positions[w] = p
        # shrink current bucket from the left
        # morally we put v at the start and then ignore it
        bucket_low[d + 1] += 1
        # move v to the end of the previous bucket
        d_new = d - 1
        high_new = bucket_high[d_new + 1]
        bucket_storage[high_new + 1] = v
        # grow previous bucket to the right
        bucket_high[d_new + 1] += 1
        # update v's stats
        degrees[v] = d_new
        positions[v] = high_new + 1
    end
    return nothing
end

function update_bucket!(
    db::DegreeBucketsColPack, v::Integer, d::Integer; degtype::Symbol, direction::Symbol
)
    (; degrees, buckets, positions) = db
    p = positions[v]
    bucket = buckets[d + 1]
    # select previous or next bucket for the move
    d_new = degree_increasing(; degtype, direction) ? d + 1 : d - 1
    bucket_new = buckets[d_new + 1]
    # put v at the end of its bucket by swapping
    w = bucket[end]
    bucket[p] = w
    bucket[end] = v
    positions[w] = p
    positions[v] = length(bucket)
    # move v from the old bucket to the new one
    pop!(bucket)
    push!(bucket_new, v)
    degrees[v] = d_new
    positions[v] = length(bucket_new)
    return nothing
end

function vertices(
    g::AdjacencyGraph{T}, ::DynamicDegreeBasedOrder{degtype,direction,reproduce_colpack}
) where {T<:Integer,degtype,direction,reproduce_colpack}
    degrees = T[degree(g, v) for v in vertices(g)]
    dmax = maximum(degrees)
    if degree_increasing(; degtype, direction)
        fill!(degrees, zero(T))
    end
    db = if reproduce_colpack
        DegreeBucketsColPack(T, degrees, dmax)
    else
        DegreeBucketsSMC(T, degrees, dmax)
    end
    nv = nb_vertices(g)
    π = Vector{T}(undef, nv)
    index_π = (direction == :low2high) ? (1:nv) : reverse(1:nv)
    degree_range = (direction == :low2high) ? reverse(0:dmax) : (0:dmax)
    for index in index_π
        u, du = pop_next_candidate!(db; degree_range)

        π[index] = u
        for v in neighbors(g, u)
            !has_diagonal(g) || (u == v && continue)
            dv = degrees[v]
            dv == -1 && continue
            update_bucket!(db, v, dv; degtype, direction)
        end
        # no need to look much further than du next time
        degree_range = if direction == :low2high
            reverse(0:min(du + 1, dmax))
        else
            max(du - 1, 0):dmax
        end
    end
    return π
end

function vertices(
    g::BipartiteGraph{T},
    ::Val{side},
    ::DynamicDegreeBasedOrder{degtype,direction,reproduce_colpack},
) where {T<:Integer,side,degtype,direction,reproduce_colpack}
    other_side = 3 - side
    # compute dist-2 degrees in an optimized way
    n = nb_vertices(g, Val(side))
    degrees = zeros(T, n)
    visited = zeros(T, n)
    for v in vertices(g, Val(side))
        for w1 in neighbors(g, Val(side), v)
            for w2 in neighbors(g, Val(other_side), w1)
                if w2 != v && visited[w2] != v
                    degrees[v] += 1
                    visited[w2] = v
                end
            end
        end
    end
    dmax = maximum(degrees)
    if degree_increasing(; degtype, direction)
        fill!(degrees, zero(T))
    end
    db = if reproduce_colpack
        DegreeBucketsColPack(T, degrees, dmax)
    else
        DegreeBucketsSMC(T, degrees, dmax)
    end
    π = Vector{T}(undef, n)
    index_π = (direction == :low2high) ? (1:n) : (n:-1:1)
    degree_range = (direction == :low2high) ? reverse(0:dmax) : (0:dmax)
    for index in index_π
        u, du = pop_next_candidate!(db; degree_range)
        π[index] = u
        for w in neighbors(g, Val(side), u)
            for v in neighbors(g, Val(other_side), w)
                if v != u && visited[v] != -u
                    # Use -u such that we don't need to fill "visited" with 0 after the computation of the dist-2 degrees
                    visited[v] = -u
                    dv = degrees[v]
                    dv == -1 && continue
                    update_bucket!(db, v, dv; degtype, direction)
                end
            end
        end
        # no need to look much further than du next time
        degree_range = if direction == :low2high
            reverse(0:min(du + 1, dmax))
        else
            max(du - 1, 0):dmax
        end
    end
    return π
end

"""
    IncidenceDegree(; reproduce_colpack=false)

Instance of [`AbstractOrder`](@ref) which sorts vertices from lowest to highest using the dynamic back degree.

# See also

- [`DynamicDegreeBasedOrder`](@ref)
"""
function IncidenceDegree(; reproduce_colpack::Bool=false)
    return DynamicDegreeBasedOrder{:back,:low2high,reproduce_colpack}()
end

"""
    SmallestLast(; reproduce_colpack=false)

Instance of [`AbstractOrder`](@ref) which sorts vertices from highest to lowest using the dynamic back degree.

# See also

- [`DynamicDegreeBasedOrder`](@ref)
"""
function SmallestLast(; reproduce_colpack::Bool=false)
    return DynamicDegreeBasedOrder{:back,:high2low,reproduce_colpack}()
end

"""
    DynamicLargestFirst(; reproduce_colpack=false)

Instance of [`AbstractOrder`](@ref) which sorts vertices from lowest to highest using the dynamic forward degree.

# See also
    
- [`DynamicDegreeBasedOrder`](@ref)
"""
function DynamicLargestFirst(; reproduce_colpack::Bool=false)
    return DynamicDegreeBasedOrder{:forward,:low2high,reproduce_colpack}()
end

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

function all_orders()
    return [
        NaturalOrder(),
        RandomOrder(),
        LargestFirst(),
        SmallestLast(),
        SmallestLast(; reproduce_colpack=true),
        IncidenceDegree(),
        IncidenceDegree(; reproduce_colpack=true),
        DynamicLargestFirst(),
        DynamicLargestFirst(; reproduce_colpack=true),
        DynamicDegreeBasedOrder{:forward,:high2low}(),
        DynamicDegreeBasedOrder{:forward,:high2low}(; reproduce_colpack=true),
    ]
end
