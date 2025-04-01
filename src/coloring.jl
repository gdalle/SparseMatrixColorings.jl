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
    star_coloring(g::AdjacencyGraph, order::AbstractOrder, postprocessing::Bool)

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
function star_coloring(g::AdjacencyGraph, order::AbstractOrder, postprocessing::Bool)
    # Initialize data structures
    nv = nb_vertices(g)
    ne = nb_edges(g)
    color = zeros(Int, nv)
    forbidden_colors = zeros(Int, nv)
    first_neighbor = fill((0, 0, 0), nv)  # at first no neighbors have been encountered
    treated = zeros(Int, nv)
    star = Vector{Int}(undef, ne)
    hub = Int[]  # one hub for each star, including the trivial ones
    vertices_in_order = vertices(g, order)

    for v in vertices_in_order
        for (w, index_vw) in neighbors_with_edge_indices(g, v)
            !has_diagonal(g) || (v == w && continue)
            iszero(color[w]) && continue
            forbidden_colors[color[w]] = v
            (p, q, _) = first_neighbor[color[w]]
            if p == v  # Case 1
                if treated[q] != v
                    # forbid colors of neighbors of q
                    _treat!(treated, forbidden_colors, g, v, q, color)
                end
                # forbid colors of neighbors of w
                _treat!(treated, forbidden_colors, g, v, w, color)
            else
                first_neighbor[color[w]] = (v, w, index_vw)
                for (x, index_wx) in neighbors_with_edge_indices(g, w)
                    !has_diagonal(g) || (w == x && continue)
                    (x == v || iszero(color[x])) && continue
                    if x == hub[star[index_wx]]  # potential Case 2 (which is always false for trivial stars with two vertices, since the associated hub is negative)
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
    star_set = StarSet(star, hub)
    if postprocessing
        # Reuse the vector forbidden_colors to compute offsets during post-processing
        offsets = forbidden_colors
        postprocess!(color, star_set, g, offsets)
    end
    return color, star_set
end

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
        !has_diagonal(g) || (w == x && continue)
        iszero(color[x]) && continue
        forbidden_colors[color[x]] = v
    end
    treated[w] = v
    return nothing
end

function _update_stars!(
    # modified
    star::AbstractVector{<:Integer},
    hub::AbstractVector{<:Integer},
    # not modified
    g::AdjacencyGraph,
    v::Integer,
    color::AbstractVector{<:Integer},
    first_neighbor::AbstractVector{<:Tuple},
)
    for (w, index_vw) in neighbors_with_edge_indices(g, v)
        !has_diagonal(g) || (v == w && continue)
        iszero(color[w]) && continue
        x_exists = false
        for (x, index_wx) in neighbors_with_edge_indices(g, w)
            !has_diagonal(g) || (w == x && continue)
            if x != v && color[x] == color[v]  # vw, wx ∈ E
                star_wx = star[index_wx]
                hub[star_wx] = w  # this may already be true
                star[index_vw] = star_wx
                x_exists = true
                break
            end
        end
        if !x_exists
            (p, q, index_pq) = first_neighbor[color[w]]
            if p == v && q != w  # vw, vq ∈ E and color[w] = color[q]
                star_vq = star[index_pq]
                hub[star_vq] = v  # this may already be true
                star[index_vw] = star_vq
            else  # vw forms a new star
                push!(hub, -max(v, w))  # star is trivial (composed only of two vertices) so we set the hub to a negative value, but it allows us to choose one of the two vertices
                star[index_vw] = length(hub)
            end
        end
    end
    return nothing
end

"""
    StarSet

Encode a set of 2-colored stars resulting from the [`star_coloring`](@ref) algorithm.

# Fields

$TYPEDFIELDS
"""
struct StarSet
    "a mapping from edges (pair of vertices) to their star index"
    star::Vector{Int}
    "a mapping from star indices to their hub (undefined hubs for single-edge stars are the negative value of one of the vertices, picked arbitrarily)"
    hub::Vector{Int}
end

"""
    acyclic_coloring(g::AdjacencyGraph, order::AbstractOrder, postprocessing::Bool)

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
function acyclic_coloring(g::AdjacencyGraph, order::AbstractOrder, postprocessing::Bool)
    # Initialize data structures
    nv = nb_vertices(g)
    ne = nb_edges(g)
    color = zeros(Int, nv)
    forbidden_colors = zeros(Int, nv)
    first_neighbor = fill((0, 0, 0), nv)  # at first no neighbors have been encountered
    first_visit_to_tree = fill((0, 0), ne)
    forest = Forest{Int}(ne)
    vertices_in_order = vertices(g, order)

    for v in vertices_in_order
        for w in neighbors(g, v)
            !has_diagonal(g) || (v == w && continue)
            iszero(color[w]) && continue
            forbidden_colors[color[w]] = v
        end
        for w in neighbors(g, v)
            !has_diagonal(g) || (v == w && continue)
            iszero(color[w]) && continue
            for (x, index_wx) in neighbors_with_edge_indices(g, w)
                !has_diagonal(g) || (w == x && continue)
                iszero(color[x]) && continue
                if forbidden_colors[color[x]] != v
                    _prevent_cycle!(
                        v,
                        w,
                        x,
                        index_wx,
                        color,
                        first_visit_to_tree,
                        forbidden_colors,
                        forest,
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
        for (w, index_vw) in neighbors_with_edge_indices(g, v)  # grow two-colored stars around the vertex v
            !has_diagonal(g) || (v == w && continue)
            iszero(color[w]) && continue
            _grow_star!(v, w, index_vw, color, first_neighbor, forest)
        end
        for (w, index_vw) in neighbors_with_edge_indices(g, v)
            !has_diagonal(g) || (v == w && continue)
            iszero(color[w]) && continue
            for (x, index_wx) in neighbors_with_edge_indices(g, w)
                !has_diagonal(g) || (w == x && continue)
                (x == v || iszero(color[x])) && continue
                if color[x] == color[v]
                    _merge_trees!(v, w, x, index_vw, index_wx, forest)  # merge trees T₁ ∋ vw and T₂ ∋ wx if T₁ != T₂
                end
            end
        end
    end

    tree_set = TreeSet(g, forest)
    if postprocessing
        # Reuse the vector forbidden_colors to compute offsets during post-processing
        offsets = forbidden_colors
        postprocess!(color, tree_set, g, offsets)
    end
    return color, tree_set
end

function _prevent_cycle!(
    # not modified
    v::Integer,
    w::Integer,
    x::Integer,
    index_wx::Integer,
    color::AbstractVector{<:Integer},
    # modified
    first_visit_to_tree::AbstractVector{<:Tuple},
    forbidden_colors::AbstractVector{<:Integer},
    forest::Forest{<:Integer},
)
    id = find_root!(forest, index_wx)  # The edge wx belongs to the 2-colored tree T, represented by an edge with an integer ID
    (p, q) = first_visit_to_tree[id]
    if p != v  # T is being visited from vertex v for the first time
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
    index_vw::Integer,
    color::AbstractVector{<:Integer},
    # modified
    first_neighbor::AbstractVector{<:Tuple},
    forest::Forest{<:Integer},
)
    # Create a new tree T_{vw} consisting only of edge vw
    (p, q, index_pq) = first_neighbor[color[w]]
    if p != v  # a neighbor of v with color[w] encountered for the first time
        first_neighbor[color[w]] = (v, w, index_vw)
    else  # merge T_{vw} with a two-colored star being grown around v
        root1 = find_root!(forest, index_vw)
        root2 = find_root!(forest, index_pq)
        root_union!(forest, root1, root2)
    end
    return nothing
end

function _merge_trees!(
    # not modified
    v::Integer,
    w::Integer,
    x::Integer,
    index_vw::Integer,
    index_wx::Integer,
    # modified
    forest::Forest{<:Integer},
)
    root1 = find_root!(forest, index_vw)
    root2 = find_root!(forest, index_wx)
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
    reverse_bfs_orders::Vector{Vector{Tuple{Int,Int}}}
    is_star::Vector{Bool}
end

function TreeSet(g::AdjacencyGraph, forest::Forest{Int})
    S = pattern(g)
    edge_to_index = edge_indices(g)
    nv = nb_vertices(g)
    nt = forest.num_trees

    # dictionary that maps a tree's root to the index of the tree
    roots = Dict{Int,Int}()
    sizehint!(roots, nt)

    # vector of dictionaries where each dictionary stores the neighbors of each vertex in a tree
    trees = [Dict{Int,Vector{Int}}() for i in 1:nt]

    # current number of roots found
    nr = 0

    rvS = rowvals(S)
    for j in axes(S, 2)
        for pos in nzrange(S, j)
            i = rvS[pos]
            if i > j
                index_ij = edge_to_index[pos]
                root = find_root!(forest, index_ij)

                # Update roots
                if !haskey(roots, root)
                    nr += 1
                    roots[root] = nr
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
        end
    end

    # degrees is a vector of integers that stores the degree of each vertex in a tree
    degrees = Vector{Int}(undef, nv)

    # reverse breadth first (BFS) traversal order for each tree in the forest
    reverse_bfs_orders = [Tuple{Int,Int}[] for i in 1:nt]

    # nvmax is the number of vertices of the biggest tree in the forest
    nvmax = 0
    for k in 1:nt
        nb_vertices_tree = length(trees[k])
        nvmax = max(nvmax, nb_vertices_tree)
    end

    # Create a queue with a fixed size nvmax
    queue = Vector{Int}(undef, nvmax)

    # Specify if each tree in the forest is a star,
    # meaning that one vertex is directly connected to all other vertices in the tree
    is_star = Vector{Bool}(undef, nt)

    for k in 1:nt
        tree = trees[k]

        # Boolean indicating whether the current tree is a star (a single central vertex connected to all others)
        bool_star = true

        # Candidate hub vertex if the current tree is a star
        virtual_hub = 0

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
                # Check if neighbor is the parent of the leaf or if it was a child before the tree was pruned
                if degrees[neighbor] != 0
                    # (leaf, neighbor) represents the next edge to visit during decompression
                    push!(reverse_bfs_orders[k], (leaf, neighbor))

                    if bool_star
                        # Initialize the potential hub of the star with the first parent of a leaf
                        if virtual_hub == 0
                            virtual_hub = neighbor
                        else
                            # Verify if the tree still qualifies as a star
                            # If we find leaves with different parents, then it can't be a star
                            if virtual_hub != neighbor
                                bool_star = false
                            end
                        end
                    end

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

        # Specify if the tree is a star or not
        is_star[k] = bool_star
    end

    return TreeSet(reverse_bfs_orders, is_star)
end

## Postprocessing, mirrors decompression code

function postprocess!(
    color::AbstractVector{<:Integer},
    star_or_tree_set::Union{StarSet,TreeSet},
    g::AdjacencyGraph,
    offsets::Vector{Int},
)
    S = pattern(g)
    edge_to_index = edge_indices(g)
    # flag which colors are actually used during decompression
    nb_colors = maximum(color)
    color_used = zeros(Bool, nb_colors)

    # nonzero diagonal coefficients force the use of their respective color (there can be no neutral colors if the diagonal is fully nonzero)
    if has_diagonal(g)
        for i in axes(S, 1)
            if !iszero(S[i, i])
                color_used[color[i]] = true
            end
        end
    end

    if star_or_tree_set isa StarSet
        # only the colors of the hubs are used
        (; star, hub) = star_or_tree_set
        nb_trivial_stars = 0

        # Iterate through all non-trivial stars
        for s in eachindex(hub)
            h = hub[s]
            if h > 0
                color_used[color[h]] = true
            else
                nb_trivial_stars += 1
            end
        end

        # Process the trivial stars (if any)
        if nb_trivial_stars > 0
            rvS = rowvals(S)
            for j in axes(S, 2)
                for k in nzrange(S, j)
                    i = rvS[k]
                    if i > j
                        index_ij = edge_to_index[k]
                        s = star[index_ij]
                        h = hub[s]
                        if h < 0
                            h = abs(h)
                            spoke = h == j ? i : j
                            if color_used[color[spoke]]
                                # Switch the hub and the spoke to possibly avoid adding one more used color
                                hub[s] = spoke
                            else
                                # Keep the current hub
                                color_used[color[h]] = true
                            end
                        end
                    end
                end
            end
        end
    else
        # only the colors of non-leaf vertices are used
        (; reverse_bfs_orders, is_star) = star_or_tree_set
        nb_trivial_trees = 0

        # Iterate through all non-trivial trees
        for k in eachindex(reverse_bfs_orders)
            reverse_bfs_order = reverse_bfs_orders[k]
            # Check if we have more than one edge in the tree (non-trivial tree)
            if length(reverse_bfs_order) > 1
                # Determine if the tree is a star
                if is_star[k]
                    # It is a non-trivial star and only the color of the hub is needed
                    (_, hub) = reverse_bfs_order[1]
                    color_used[color[hub]] = true
                else
                    # It is not a star and both colors are needed during the decompression
                    (i, j) = reverse_bfs_order[1]
                    color_used[color[i]] = true
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
        num_colors_useless = 0

        # determine what are the useless colors and compute the offsets
        for ci in 1:nb_colors
            if color_used[ci]
                offsets[ci] = num_colors_useless
            else
                num_colors_useless += 1
            end
        end

        # assign the neutral color to every vertex with a useless color and remap the colors
        for i in eachindex(color)
            ci = color[i]
            if !color_used[ci]
                # assign the neutral color
                color[i] = 0
            else
                # remap the color to not have any gap
                color[i] -= offsets[ci]
            end
        end
    end
    return color
end
