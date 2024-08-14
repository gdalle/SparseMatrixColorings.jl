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
    nvertices = length(g)
    color = zeros(Int, nvertices)
    forbidden_colors = zeros(Int, nvertices)
    first_neighbor = fill((0, 0), nvertices)  # at first no neighbors have been encountered
    treated = zeros(Int, nvertices)
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

Encode a set of 2-colored stars resulting from the [`star_coloring`](@ref) algorithm.

# Fields

$TYPEDFIELDS
"""
struct StarSet
    "a mapping from edges (pair of vertices) to their star index"
    star::Dict{Tuple{Int,Int},Int}
    "a mapping from star indices to their hub"
    hub::Vector{Int}
    "a mapping from star indices to the vector of their spokes"
    spokes::Vector{Vector{Int}}
end

function StarSet(star, hub)
    spokes = [Int[] for s in eachindex(hub)]
    for ((i, j), s) in pairs(star)
        h = hub[s]
        if i == h
            push!(spokes[s], j)
        elseif j == h
            push!(spokes[s], i)
        end
    end
    return StarSet(star, hub, spokes)
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
                push!(hub, max(v, w))  # hub is yet undefined so we can pick either vertex
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
        star_set::StarSet
    )

Return the indices `(k, c)` such that `A[i, j] = B[k, c]`, where `A` is the uncompressed symmetric matrix and `B` is the column-compressed matrix.

This function corresponds to algorithm `DirectRecover2` in the paper.

# References

> [_Efficient Computation of Sparse Hessians Using Coloring and Automatic Differentiation_](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.1080.0286), Gebremedhin et al. (2009), Figure 3
"""
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

"""
    acyclic_coloring(g::Graph, order::AbstractOrder)

Compute an acyclic coloring of all vertices in the adjacency graph `g` and return a tuple `(color, tree_set)`, where

- `color` is the vector of integer colors
- `tree_set` is a [`TreeSet`](@ref) encoding the set of 2-colored trees

An _acyclic coloring_ is a distance-1 coloring with the further restriction that every cycle uses at least 3 colors.

The vertices are colored in a greedy fashion, following the `order` supplied.

# See also

- [`Graph`](@ref)
- [`AbstractOrder`](@ref)

# References

> [_New Acyclic and Star Coloring Algorithms with Application to Computing Hessians_](https://epubs.siam.org/doi/abs/10.1137/050639879), Gebremedhin et al. (2007), Algorithm 3.1
"""
function acyclic_coloring(g::Graph, order::AbstractOrder)
    # Initialize data structures
    nvertices = length(g)
    nedges = nnz(g) ÷ 2  # symmetric sparse matrix with empty diagonal
    color = zeros(Int, nvertices)
    forbidden_colors = zeros(Int, nvertices)
    first_neighbor = fill((0, 0), nvertices)  # at first no neighbors have been encountered
    first_visit_to_tree = fill((0, 0), nedges)
    forest = DisjointSets{Tuple{Int,Int}}()
    vertices_in_order = vertices(g, order)

    for v in vertices_in_order
        for w in neighbors(g, v)
            iszero(color[w]) && continue
            forbidden_colors[color[w]] = v
        end
        for w in neighbors(g, v)
            iszero(color[w]) && continue
            for x in neighbors(g, w)
                iszero(color[x]) && continue
                if forbidden_colors[color[x]] != v
                    _prevent_cycle!(
                        v, w, x, color, first_visit_to_tree, forbidden_colors, forest
                    )
                end
            end
        end
        for i in eachindex(forbidden_colors)
            if forbidden_colors[i] != v
                color[v] = i
                break
            end
        end
        for w in neighbors(g, v)  # grow two-colored stars around the vertex v
            iszero(color[w]) && continue
            _grow_star!(v, w, color, first_neighbor, forest)
        end
        for w in neighbors(g, v)
            iszero(color[w]) && continue
            for x in neighbors(g, w)
                (x == v || iszero(color[x])) && continue
                if color[x] == color[v]
                    _merge_trees!(v, w, x, forest)  # merge trees T₁ ∋ vw and T₂ ∋ wx if T₁ != T₂
                end
            end
        end
    end

    # compress forest
    for edge in forest.revmap
        find_root!(forest, edge)
    end

    return color, TreeSet(forest)
end

function _prevent_cycle!(
    # not modified
    v::Integer,
    w::Integer,
    x::Integer,
    color::AbstractVector{<:Integer},
    # modified
    first_visit_to_tree::AbstractVector{<:Tuple},
    forbidden_colors::AbstractVector{<:Integer},
    forest::DisjointSets{<:Tuple{Int,Int}},
)
    wx = _sort(w, x)
    root = find_root!(forest, wx)  # edge wx belongs to the 2-colored tree T represented by edge "root"
    id = forest.intmap[root] # ID of the representative edge "root" of a two-colored tree T.
    (p, q) = first_visit_to_tree[id]
    if p != v  # T is being visited from vertex v for the first time
        vw = _sort(v, w)
        first_visit_to_tree[id] = (v, w)
    elseif q != w  # T is connected to vertex v via at least two edges
        forbidden_colors[color[x]] = v
    end
    return nothing
end

function _grow_star!(
    # not modified
    v::Integer,
    w::Integer,
    color::AbstractVector{<:Integer},
    # modified
    first_neighbor::AbstractVector{<:Tuple},
    forest::DisjointSets{Tuple{Int,Int}},
)
    vw = _sort(v, w)
    push!(forest, vw)  # Create a new tree T_{vw} consisting only of edge vw
    (p, q) = first_neighbor[color[w]]
    if p != v  # a neighbor of v with color[w] encountered for the first time
        first_neighbor[color[w]] = (v, w)
    else  # merge T_{vw} with a two-colored star being grown around v
        vw = _sort(v, w)
        pq = _sort(p, q)
        root1 = find_root!(forest, vw)
        root2 = find_root!(forest, pq)
        root_union!(forest, root1, root2)
    end
    return nothing
end

function _merge_trees!(
    # not modified
    v::Integer,
    w::Integer,
    x::Integer,
    # modified
    forest::DisjointSets{Tuple{Int,Int}},
)
    vw = _sort(v, w)
    wx = _sort(w, x)
    root1 = find_root!(forest, vw)
    root2 = find_root!(forest, wx)
    if root1 != root2
        root_union!(forest, root1, root2)
    end
    return nothing
end

"""
    TreeSet

Encode a set of 2-colored trees resulting from the [`acyclic_coloring`](@ref) algorithm.

# Fields

$TYPEDFIELDS
"""
struct TreeSet
    "a forest of two-colored trees"
    forest::DisjointSets{Tuple{Int,Int}}
end
