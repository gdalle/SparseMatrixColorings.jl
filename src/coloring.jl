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
    color = Vector{Int}(undef, nb_vertices(bg, Val(side)))
    forbidden_colors = Vector{Int}(undef, nb_vertices(bg, Val(side)))
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
    star_coloring(g::AdjacencyGraph, order::AbstractOrder; postprocessing::Bool)

Compute a star coloring of all vertices in the adjacency graph `g` and return a tuple `(color, star_set)`, where

- `color` is the vector of integer colors
- `star_set` is a [`StarSet`](@ref) encoding the set of 2-colored stars

A _star coloring_ is a distance-1 coloring such that every path on 4 vertices uses at least 3 colors.

The vertices are colored in a greedy fashion, following the `order` supplied.

If `postprocessing=true`, some colors might be replaced with `0` (the "neutral" color) as long as they are not needed during decompression.

# See also

- [`AdjacencyGraph`](@ref)
- [`AbstractOrder`](@ref)

# References

> [_New Acyclic and Star Coloring Algorithms with Application to Computing Hessians_](https://epubs.siam.org/doi/abs/10.1137/050639879), Gebremedhin et al. (2007), Algorithm 4.1
"""
function star_coloring(g::AdjacencyGraph, order::AbstractOrder; postprocessing::Bool)
    # Initialize data structures
    nv = nb_vertices(g)
    ne = nb_edges(g)
    color = zeros(Int, nv)
    forbidden_colors = zeros(Int, nv)
    first_neighbor = fill((0, 0), nv)  # at first no neighbors have been encountered
    treated = zeros(Int, nv)
    star = Dict{Tuple{Int,Int},Int}()
    sizehint!(star, ne)
    hub = Int[]  # one hub for each star, including the trivial ones
    nb_spokes = Int[]  # number of spokes for each star
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
                    if x == hub[star[wx]]  # potential Case 2 (which is always false for trivial stars with two vertices, since the associated hub is negative)
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
        _update_stars!(star, hub, nb_spokes, g, v, color, first_neighbor)
    end
    star_set = StarSet(star, hub, nb_spokes)
    if postprocessing
        postprocess!(color, star_set, g)
    end
    return color, star_set
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
    "a mapping from star indices to their hub (undefined hubs for single-edge stars are the negative value of one of the vertices, picked arbitrarily)"
    hub::Vector{Int}
    "a mapping from star indices to the vector of their spokes"
    spokes::Vector{Vector{Int}}
end

function StarSet(star::Dict{Tuple{Int,Int},Int}, hub::Vector{Int}, nb_spokes::Vector{Int})
    # Create a list of spokes for each star, preallocating their sizes based on nb_spokes
    spokes = [Vector{Int}(undef, ns) for ns in nb_spokes]

    # Reuse nb_spokes as counters to track the current index while filling the spokes
    fill!(nb_spokes, 0)

    for ((i, j), s) in pairs(star)
        h = abs(hub[s])
        nb_spokes[s] += 1
        index = nb_spokes[s]

        # Assign the non-hub vertex (spoke) to the correct position in spokes
        if i == h
            spokes[s][index] = j
        elseif j == h
            spokes[s][index] = i
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
    g::AdjacencyGraph,
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
    nb_spokes::AbstractVector{<:Integer},
    # not modified
    g::AdjacencyGraph,
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
                star_wx = star[wx]
                hub[star_wx] = w  # this may already be true
                nb_spokes[star_wx] += 1
                star[vw] = star_wx
                x_exists = true
                break
            end
        end
        if !x_exists
            (p, q) = first_neighbor[color[w]]
            if p == v && q != w  # vw, vq ∈ E and color[w] = color[q]
                vq = _sort(v, q)
                star_vq = star[vq]
                hub[star_vq] = v  # this may already be true
                nb_spokes[star_vq] += 1
                star[vw] = star_vq
            else  # vw forms a new star
                push!(hub, -max(v, w))  # star is trivial (composed only of two vertices) so we set the hub to a negative value, but it allows us to choose one of the two vertices
                push!(nb_spokes, 1)
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
    (; star, hub) = star_set
    if i == j
        # diagonal
        return i, color[j]
    end
    if i > j  # keys of star are sorted tuples
        # star only contains one triangle
        i, j = j, i
    end
    star_id = star[i, j]
    h = abs(hub[star_id])
    if h == j
        # i is the spoke
        return i, color[h]
    else
        # j is the spoke
        return j, color[h]
    end
end

"""
    acyclic_coloring(g::AdjacencyGraph, order::AbstractOrder; postprocessing::Bool)

Compute an acyclic coloring of all vertices in the adjacency graph `g` and return a tuple `(color, tree_set)`, where

- `color` is the vector of integer colors
- `tree_set` is a [`TreeSet`](@ref) encoding the set of 2-colored trees

An _acyclic coloring_ is a distance-1 coloring with the further restriction that every cycle uses at least 3 colors.

The vertices are colored in a greedy fashion, following the `order` supplied.

If `postprocessing=true`, some colors might be replaced with `0` (the "neutral" color) as long as they are not needed during decompression.

# See also

- [`AdjacencyGraph`](@ref)
- [`AbstractOrder`](@ref)

# References

> [_New Acyclic and Star Coloring Algorithms with Application to Computing Hessians_](https://epubs.siam.org/doi/abs/10.1137/050639879), Gebremedhin et al. (2007), Algorithm 3.1
"""
function acyclic_coloring(g::AdjacencyGraph, order::AbstractOrder; postprocessing::Bool)
    # Initialize data structures
    nv = nb_vertices(g)
    ne = nb_edges(g)
    color = zeros(Int, nv)
    forbidden_colors = zeros(Int, nv)
    first_neighbor = fill((0, 0), nv)  # at first no neighbors have been encountered
    first_visit_to_tree = fill((0, 0), ne)
    forest = DisjointSets{Tuple{Int,Int}}()
    sizehint!(forest.intmap, ne)
    sizehint!(forest.revmap, ne)
    sizehint!(forest.internal.parents, ne)
    sizehint!(forest.internal.ranks, ne)
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
    tree_set = TreeSet(forest, nb_vertices(g))
    if postprocessing
        postprocess!(color, tree_set, g)
    end
    return color, tree_set
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
    vertices_by_tree::Vector{Vector{Int}}
    reverse_bfs_orders::Vector{Vector{Tuple{Int,Int}}}
end

function TreeSet(forest::DisjointSets{Tuple{Int,Int}}, nvertices::Int)
    # forest is a structure DisjointSets from DataStructures.jl
    # - forest.intmap: a dictionary that maps an edge (i, j) to an integer k
    # - forest.revmap: a dictionary that does the reverse of intmap, mapping an integer k to an edge (i, j)
    # - forest.internal.ngroups: the number of trees in the forest
    ntrees = forest.internal.ngroups

    # dictionary that maps a tree's root to the index of the tree
    roots = Dict{Int,Int}()

    # vector of dictionaries where each dictionary stores the neighbors of each vertex in a tree
    trees = [Dict{Int,Vector{Int}}() for i in 1:ntrees]

    # counter of the number of roots found
    k = 0
    for edge in forest.revmap
        i, j = edge
        # forest has already been compressed so this doesn't change its state
        # I wanted to use find_root but it is deprecated
        root_edge = find_root!(forest, edge)
        root = forest.intmap[root_edge]

        # Update roots
        if !haskey(roots, root)
            k += 1
            roots[root] = k
        end

        # index of the tree T that contains this edge
        index_tree = roots[root]

        # Update the neighbors of i in the tree T
        if !haskey(trees[index_tree], i)
            trees[index_tree][i] = [j]
        else
            push!(trees[index_tree][i], j)
        end

        # Update the neighbors of j in the tree T
        if !haskey(trees[index_tree], j)
            trees[index_tree][j] = [i]
        else
            push!(trees[index_tree][j], i)
        end
    end

    # degrees is a vector of integers that stores the degree of each vertex in a tree
    degrees = Vector{Int}(undef, nvertices)

    # list of vertices for each tree in the forest
    vertices_by_tree = [collect(keys(trees[i])) for i in 1:ntrees]

    # reverse breadth first (BFS) traversal order for each tree in the forest
    reverse_bfs_orders = [Tuple{Int,Int}[] for i in 1:ntrees]

    # nvmax is the number of vertices of the biggest tree in the forest
    nvmax = mapreduce(length, max, vertices_by_tree; init=0)

    # Create a queue with a fixed size nvmax
    queue = Vector{Int}(undef, nvmax)

    for k in 1:ntrees
        tree = trees[k]

        # Initialize the queue to store the leaves
        queue_start = 1
        queue_end = 0

        # compute the degree of each vertex in the tree
        for (vertex, neighbors) in tree
            degree = length(neighbors)
            degrees[vertex] = degree

            # the vertex is a leaf
            if degree == 1
                queue_end += 1
                queue[queue_end] = vertex
            end
        end

        # continue until all leaves are treated
        while queue_start <= queue_end
            leaf = queue[queue_start]
            queue_start += 1

            # Mark the vertex as removed
            degrees[leaf] = 0

            for neighbor in tree[leaf]
                if degrees[neighbor] != 0
                    # (leaf, neighbor) represents the next edge to visit during decompression
                    push!(reverse_bfs_orders[k], (leaf, neighbor))

                    # reduce the degree of the neighbor
                    degrees[neighbor] -= 1

                    # check if the neighbor is now a leaf
                    if degrees[neighbor] == 1
                        queue_end += 1
                        queue[queue_end] = neighbor
                    end
                end
            end
        end
    end

    return TreeSet(vertices_by_tree, reverse_bfs_orders)
end

## Postprocessing, mirrors decompression code

function postprocess!(
    color::AbstractVector{<:Integer},
    star_or_tree_set::Union{StarSet,TreeSet},
    g::AdjacencyGraph,
)
    (; S, has_loops) = g
    # flag which colors are actually used during decompression
    color_used = zeros(Bool, maximum(color))

    # nonzero diagonal coefficients force the use of their respective color (there can be no neutral colors if the diagonal is fully nonzero)
    if has_loops
        for i in axes(S, 1)
            if !iszero(S[i, i])
                color_used[color[i]] = true
            end
        end
    end

    if star_or_tree_set isa StarSet
        # only the colors of the hubs are used
        (; hub, spokes) = star_or_tree_set
        nb_trivial_stars = 0

        # Iterate through all non-trivial stars
        for s in eachindex(hub)
            j = hub[s]
            if j > 0
                color_used[color[j]] = true
            else
                nb_trivial_stars += 1
            end
        end

        # Process the trivial stars (if any)
        if nb_trivial_stars > 0
            for s in eachindex(hub)
                j = hub[s]
                if j < 0
                    i = spokes[s][1]
                    j = abs(j)
                    if color_used[color[i]]
                        # Make i the hub to avoid possibly adding one more used color
                        # Switch it with the (only) spoke
                        hub[s] = i
                        spokes[s][1] = j
                    else
                        # Keep j as the hub
                        color_used[color[j]] = true
                    end
                end
            end
        end
    else
        # only the colors of non-leaf vertices are used
        (; reverse_bfs_orders) = star_or_tree_set
        nb_trivial_trees = 0

        # Iterate through all non-trivial trees
        for k in eachindex(reverse_bfs_orders)
            reverse_bfs_order = reverse_bfs_orders[k]
            # Check if we have more than one edge in the tree
            if length(reverse_bfs_order) > 1
                # TODO: Optimize by avoiding iteration over all edges
                # Only one edge is needed if we know if it is a normal tree or a star
                for (i, j) in reverse_bfs_order
                    color_used[color[j]] = true
                end
            else
                nb_trivial_trees += 1
            end
        end

        # Process the trivial trees (if any)
        if nb_trivial_trees > 0
            for k in eachindex(reverse_bfs_orders)
                reverse_bfs_order = reverse_bfs_orders[k]
                # Check if we have exactly one edge in the tree
                if length(reverse_bfs_order) == 1
                    (i, j) = reverse_bfs_order[1]
                    if color_used[color[i]]
                        # Make i the root to avoid possibly adding one more used color
                        # Switch it with the (only) leaf
                        reverse_bfs_order[1] = (j, i)
                    else
                        # Keep j as the root
                        color_used[color[j]] = true
                    end
                end
            end
        end
    end

    # if at least one of the colors is useless, modify the color assignments of vertices
    if any(!, color_used)
        # assign the neutral color to every vertex with a useless color
        for i in eachindex(color)
            ci = color[i]
            if !color_used[ci]
                color[i] = 0
            end
        end
        # remap colors to decrease the highest one by filling gaps
        color .= remap_colors(color)[1]
    end
    return color
end
