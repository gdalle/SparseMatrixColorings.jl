"""
    partial_distance2_coloring(bg::BipartiteGraph, ::Val{side}, order::AbstractOrder)

Compute a distance-2 coloring of the given `side` (`1` or `2`) in the bipartite graph `bg` and return a vector of integer colors.

A _distance-2 coloring_ is such that two vertices have different colors if they are at distance at most 2.

The vertices are colored in a greedy fashion, following the `order` supplied.

# See also

- [`BipartiteGraph`](@ref)
- [`AbstractOrder`](@ref)

# References

> [_What Color Is Your Jacobian? Graph Coloring for Computing Derivatives_](https://epubs.siam.org/doi/10.1137/S0036144504444711), Gebremedhin et al. (2005), Algorithm 3.2
"""
function partial_distance2_coloring(
    bg::BipartiteGraph, ::Val{side}, order::AbstractOrder
) where {side}
    color = Vector{Int}(undef, length(bg, Val(side)))
    forbidden_colors = Vector{Int}(undef, length(bg, Val(side)))
    vertices_in_order = vertices(bg, Val(side), order)
    partial_distance2_coloring!(color, forbidden_colors, bg, Val(side), vertices_in_order)
    return color
end

function partial_distance2_coloring!(
    color::Vector{Int},
    forbidden_colors::Vector{Int},
    bg::BipartiteGraph,
    ::Val{side},
    vertices_in_order::AbstractVector{<:Integer},
) where {side}
    color .= 0
    forbidden_colors .= 0
    other_side = 3 - side
    for v in vertices_in_order
        for w in neighbors(bg, Val(side), v)
            for x in neighbors(bg, Val(other_side), w)
                if !iszero(color[x])
                    forbidden_colors[color[x]] = v
                end
            end
        end
        for i in eachindex(forbidden_colors)
            if forbidden_colors[i] != v
                color[v] = i
                break
            end
        end
    end
end

"""
    star_coloring(g::Graph, order::AbstractOrder)

Compute a star coloring of all vertices in the adjacency graph `g` and return a tuple `(color, star_set)`, where

- `color` is the vector of integer colors
- `star_set` is a [`StarSet`](@ref) encoding the set of 2-colored stars

A _star coloring_ is a distance-1 coloring such that every path on 4 vertices uses at least 3 colors.

The vertices are colored in a greedy fashion, following the `order` supplied.

# See also

- [`Graph`](@ref)
- [`AbstractOrder`](@ref)

# References

> [_New Acyclic and Star Coloring Algorithms with Application to Computing Hessians_](https://epubs.siam.org/doi/abs/10.1137/050639879), Gebremedhin et al. (2007), Algorithm 4.1
"""
function star_coloring(g::Graph, order::AbstractOrder)
    # Initialize data structures
    color = zeros(Int, length(g))
    forbidden_colors = zeros(Int, length(g))
    first_neighbor = fill((0, 0), length(g))  # at first no neighbors have been encountered
    treated = zeros(Int, length(g))
    star = Dict{Tuple{Int,Int},Int}()
    hub = Int[]
    vertices_in_order = vertices(g, order)

    for v in vertices_in_order
        for w in neighbors(g, v)
            iszero(color[w]) && continue
            forbidden_colors[color[w]] = v
            (p, q) = first_neighbor[color[w]]
            if p == v  # Case 1
                if treated[q] != v
                    # forbid colors of neighbors of q
                    _treat!(treated, forbidden_colors, g, v, q, color)
                end
                # forbid colors of neighbors of w
                _treat!(treated, forbidden_colors, g, v, w, color)
            else
                first_neighbor[color[w]] = (v, w)
                for x in neighbors(g, w)
                    (x == v || iszero(color[x])) && continue
                    wx = _sort(w, x)
                    if x == hub[star[wx]]  # potential Case 2
                        forbidden_colors[color[x]] = v
                    end
                end
            end
        end
        for i in eachindex(forbidden_colors)
            if forbidden_colors[i] != v
                color[v] = i
                break
            end
        end
        _update_stars!(star, hub, g, v, color, first_neighbor)
    end
    return color, StarSet(star, hub)
end

"""
    StarSet

Encode a set of 2-colored stars resulting from the star coloring algorithm.

# Fields

The fields are not part of the public API, even though the type is.

- `star::Dict{Tuple{Int,Int},Int}`: a mapping from edges (pair of vertices) their to star index
- `hub::Vector{Int}`: a mapping from star indices to their hub (the hub is `0` if the star only contains one edge)

# References

> [_New Acyclic and Star Coloring Algorithms with Application to Computing Hessians_](https://epubs.siam.org/doi/abs/10.1137/050639879), Gebremedhin et al. (2007), Algorithm 4.1
"""
struct StarSet
    star::Dict{Tuple{Int,Int},Int}
    hub::Vector{Int}
end

_sort(u, v) = (min(u, v), max(u, v))

function _treat!(
    # modified
    treated::AbstractVector{<:Integer},
    forbidden_colors::AbstractVector{<:Integer},
    # not modified
    g::Graph,
    v::Integer,
    w::Integer,
    color::AbstractVector{<:Integer},
)
    for x in neighbors(g, w)
        iszero(color[x]) && continue
        forbidden_colors[color[x]] = v
    end
    treated[w] = v
    return nothing
end

function _update_stars!(
    # modified
    star::Dict{<:Tuple,<:Integer},
    hub::AbstractVector{<:Integer},
    # not modified
    g::Graph,
    v::Integer,
    color::AbstractVector{<:Integer},
    first_neighbor::AbstractVector{<:Tuple},
)
    for w in neighbors(g, v)
        iszero(color[w]) && continue
        vw = _sort(v, w)
        x_exists = false
        for x in neighbors(g, w)
            if x != v && color[x] == color[v]  # vw, wx ∈ E
                wx = _sort(w, x)
                hub[star[wx]] = w  # this may already be true
                star[vw] = star[wx]
                x_exists = true
                break
            end
        end
        if !x_exists
            (p, q) = first_neighbor[color[w]]
            if p == v && q != w  # vw, vq ∈ E and color[w] = color[q]
                vq = _sort(v, q)
                hub[star[vq]] = v  # this may already be true
                star[vw] = star[vq]
            else  # vw forms a new star
                push!(hub, 0)  # hub is yet undefined
                star[vw] = length(hub)
            end
        end
    end
    return nothing
end

"""
    symmetric_coefficient(
        i::Integer, j::Integer,
        color::AbstractVector{<:Integer},
        group::AbstractVector{<:AbstractVector{<:Integer}},
        S::AbstractMatrix{Bool}
    )

    symmetric_coefficient(
        i::Integer, j::Integer,
        color::AbstractVector{<:Integer},
        star_set::StarSet
    )

Return the indices `(k, c)` such that `A[i, j] = B[k, c]`, where `A` is the uncompressed symmetric matrix and `B` is the column-compressed matrix.

The first version corresponds to algorithm `DirectRecover1` in the paper, the second to `DirectRecover2`.

# References

> [_Efficient Computation of Sparse Hessians Using Coloring and Automatic Differentiation_](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.1080.0286), Gebremedhin et al. (2009), Figures 2 and 3
"""
function symmetric_coefficient end

function symmetric_coefficient(
    i::Integer,
    j::Integer,
    color::AbstractVector{<:Integer},
    group::AbstractVector{<:AbstractVector{<:Integer}},
    S::AbstractMatrix,
)
    for j2 in group[color[j]]
        j2 == j && continue
        if !iszero(S[i, j2])
            return j, color[i]
        end
    end
    return i, color[j]
end

function symmetric_coefficient(
    i::Integer, j::Integer, color::AbstractVector{<:Integer}, star_set::StarSet
)
    @compat (; star, hub) = star_set
    if i == j
        # diagonal
        return i, color[j]
    end
    if i > j  # keys of star are sorted tuples
        # star only contains one triangle
        i, j = j, i
    end
    star_id = star[i, j]
    h = hub[star_id]
    if h == 0
        # pick arbitrary hub
        h = i
    end
    if h == j
        # i is the spoke
        return i, color[h]
    elseif h == i
        # j is the spoke
        return j, color[h]
    end
end
