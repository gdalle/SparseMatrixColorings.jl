struct InvalidColoringError <: Exception end

"""
    partial_distance2_coloring(
        bg::BipartiteGraph, ::Val{side}, vertices_in_order::AbstractVector;
        forced_colors::Union{AbstractVector{<:Integer},Nothing}=nothing
    )

Compute a distance-2 coloring of the given `side` (`1` or `2`) in the bipartite graph `bg` and return a vector of integer colors.

A _distance-2 coloring_ is such that two vertices have different colors if they are at distance at most 2.

The vertices are colored in a greedy fashion, following the order supplied.

The optional `forced_colors` keyword argument is used to enforce predefined vertex colors (e.g. coming from another optimization algorithm) but still run the distance-2 coloring procedure to verify correctness.

# See also

- [`BipartiteGraph`](@ref)
- [`AbstractOrder`](@ref)

# References

> [_What Color Is Your Jacobian? Graph Coloring for Computing Derivatives_](https://epubs.siam.org/doi/10.1137/S0036144504444711), Gebremedhin et al. (2005), Algorithm 3.2
"""
function partial_distance2_coloring(
    bg::BipartiteGraph{T},
    ::Val{side},
    vertices_in_order::AbstractVector{<:Integer};
    forced_colors::Union{AbstractVector{<:Integer},Nothing}=nothing,
) where {T,side}
    color = Vector{T}(undef, nb_vertices(bg, Val(side)))
    forbidden_colors = Vector{T}(undef, nb_vertices(bg, Val(side)))
    partial_distance2_coloring!(
        color, forbidden_colors, bg, Val(side), vertices_in_order; forced_colors
    )
    return color
end

function partial_distance2_coloring!(
    color::AbstractVector{<:Integer},
    forbidden_colors::AbstractVector{<:Integer},
    bg::BipartiteGraph,
    ::Val{side},
    vertices_in_order::AbstractVector{<:Integer};
    forced_colors::Union{AbstractVector{<:Integer},Nothing}=nothing,
) where {side}
    color .= 0
    forbidden_colors .= 0
    other_side = 3 - side
    for v in vertices_in_order
        for w in neighbors(bg, Val(side), v)
            for x in neighbors(bg, Val(other_side), w)
                color_x = color[x]
                if !iszero(color_x)
                    forbidden_colors[color_x] = v
                end
            end
        end
        if isnothing(forced_colors)
            for i in eachindex(forbidden_colors)
                if forbidden_colors[i] != v
                    color[v] = i
                    break
                end
            end
        else
            f = forced_colors[v]
            if (
                (f == 0 && length(neighbors(bg, Val(side), v)) > 0) ||
                (f > 0 && forbidden_colors[f] == v)
            )
                throw(InvalidColoringError())
            else
                color[v] = f
            end
        end
    end
end

"""
    star_coloring(
        g::AdjacencyGraph, vertices_in_order::AbstractVector, postprocessing::Bool;
        forced_colors::Union{AbstractVector,Nothing}=nothing
    )

Compute a star coloring of all vertices in the adjacency graph `g` and return a tuple `(color, star_set)`, where

- `color` is the vector of integer colors
- `star_set` is a [`StarSet`](@ref) encoding the set of 2-colored stars

A _star coloring_ is a distance-1 coloring such that every path on 4 vertices uses at least 3 colors.

The vertices are colored in a greedy fashion, following the order supplied.

If `postprocessing=true`, some colors might be replaced with `0` (the "neutral" color) as long as they are not needed during decompression.

The optional `forced_colors` keyword argument is used to enforce predefined vertex colors (e.g. coming from another optimization algorithm) but still run the star coloring procedure to verify correctness and build auxiliary data structures, useful during decompression.

# See also

- [`AdjacencyGraph`](@ref)
- [`AbstractOrder`](@ref)

# References

> [_New Acyclic and Star Coloring Algorithms with Application to Computing Hessians_](https://epubs.siam.org/doi/abs/10.1137/050639879), Gebremedhin et al. (2007), Algorithm 4.1
"""
function star_coloring(
    g::AdjacencyGraph{T},
    vertices_in_order::AbstractVector{<:Integer},
    postprocessing::Bool;
    forced_colors::Union{AbstractVector{<:Integer},Nothing}=nothing,
) where {T<:Integer}
    # Initialize data structures
    nv = nb_vertices(g)
    ne = nb_edges(g)
    color = zeros(T, nv)
    forbidden_colors = zeros(T, nv)
    first_neighbor = fill((zero(T), zero(T), zero(T)), nv)  # at first no neighbors have been encountered
    treated = zeros(T, nv)
    star = Vector{T}(undef, ne)
    hub = T[]  # one hub for each star, including the trivial ones

    for v in vertices_in_order
        for (w, index_vw) in neighbors_with_edge_indices(g, v)
            augmented_graph(g) || (v == w && continue)
            color_w = color[w]
            iszero(color_w) && continue
            forbidden_colors[color_w] = v
            (p, q, _) = first_neighbor[color_w]
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
                    augmented_graph(g) || (w == x && continue)
                    color_x = color[x]
                    (x == v || iszero(color_x)) && continue
                    if x == hub[star[index_wx]]  # potential Case 2 (which is always false for trivial stars with two vertices, since the associated hub is negative)
                        forbidden_colors[color_x] = v
                    end
                end
            end
        end
        if isnothing(forced_colors)
            for i in eachindex(forbidden_colors)
                if forbidden_colors[i] != v
                    color[v] = i
                    break
                end
            end
        else
            if forbidden_colors[forced_colors[v]] == v  # TODO: handle forced_colors[v] == 0
                throw(InvalidColoringError())
            else
                color[v] = forced_colors[v]
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
        augmented_graph(g) || (w == x && continue)
        color_x = color[x]
        iszero(color_x) && continue
        forbidden_colors[color_x] = v
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
        augmented_graph(g) || (v == w && continue)
        color_w = color[w]
        iszero(color_w) && continue
        x_exists = false
        for (x, index_wx) in neighbors_with_edge_indices(g, w)
            augmented_graph(g) || (w == x && continue)
            if x != v && color[x] == color[v]  # vw, wx ∈ E
                star_wx = star[index_wx]
                hub[star_wx] = w  # this may already be true
                star[index_vw] = star_wx
                x_exists = true
                break
            end
        end
        if !x_exists
            (p, q, index_pq) = first_neighbor[color_w]
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
struct StarSet{T}
    "a mapping from edges (pair of vertices) to their star index"
    star::Vector{T}
    "a mapping from star indices to their hub (undefined hubs for single-edge stars are the negative value of one of the vertices, picked arbitrarily)"
    hub::Vector{T}
end

"""
    acyclic_coloring(g::AdjacencyGraph, vertices_in_order::AbstractVector, postprocessing::Bool)

Compute an acyclic coloring of all vertices in the adjacency graph `g` and return a tuple `(color, tree_set)`, where

- `color` is the vector of integer colors
- `tree_set` is a [`TreeSet`](@ref) encoding the set of 2-colored trees

An _acyclic coloring_ is a distance-1 coloring with the further restriction that every cycle uses at least 3 colors.

The vertices are colored in a greedy fashion, following the order supplied.

If `postprocessing=true`, some colors might be replaced with `0` (the "neutral" color) as long as they are not needed during decompression.

# See also

- [`AdjacencyGraph`](@ref)
- [`AbstractOrder`](@ref)

# References

> [_New Acyclic and Star Coloring Algorithms with Application to Computing Hessians_](https://epubs.siam.org/doi/abs/10.1137/050639879), Gebremedhin et al. (2007), Algorithm 3.1
"""
function acyclic_coloring(
    g::AdjacencyGraph{T}, vertices_in_order::AbstractVector{<:Integer}, postprocessing::Bool
) where {T<:Integer}
    # Initialize data structures
    nv = nb_vertices(g)
    ne = nb_edges(g)
    color = zeros(T, nv)
    forbidden_colors = zeros(T, nv)
    first_neighbor = fill((zero(T), zero(T), zero(T)), nv)  # at first no neighbors have been encountered
    first_visit_to_tree = fill((zero(T), zero(T)), ne)
    forest = Forest{T}(ne)

    for v in vertices_in_order
        for w in neighbors(g, v)
            augmented_graph(g) || (v == w && continue)
            color_w = color[w]
            iszero(color_w) && continue
            forbidden_colors[color_w] = v
        end
        for w in neighbors(g, v)
            augmented_graph(g) || (v == w && continue)
            iszero(color[w]) && continue
            for (x, index_wx) in neighbors_with_edge_indices(g, w)
                augmented_graph(g) || (w == x && continue)
                color_x = color[x]
                iszero(color_x) && continue
                if forbidden_colors[color_x] != v
                    _prevent_cycle!(
                        v,
                        w,
                        x,
                        index_wx,
                        color_x,
                        first_visit_to_tree,
                        forbidden_colors,
                        forest,
                    )
                end
            end
        end
        # TODO: handle forced colors
        for i in eachindex(forbidden_colors)
            if forbidden_colors[i] != v
                color[v] = i
                break
            end
        end
        for (w, index_vw) in neighbors_with_edge_indices(g, v)  # grow two-colored stars around the vertex v
            augmented_graph(g) || (v == w && continue)
            color_w = color[w]
            iszero(color_w) && continue
            _grow_star!(v, w, index_vw, color_w, first_neighbor, forest)
        end
        for (w, index_vw) in neighbors_with_edge_indices(g, v)
            augmented_graph(g) || (v == w && continue)
            iszero(color[w]) && continue
            for (x, index_wx) in neighbors_with_edge_indices(g, w)
                augmented_graph(g) || (w == x && continue)
                color_x = color[x]
                (x == v || iszero(color_x)) && continue
                if color_x == color[v]
                    _merge_trees!(v, w, x, index_vw, index_wx, forest)  # merge trees T₁ ∋ vw and T₂ ∋ wx if T₁ != T₂
                end
            end
        end
    end

    buffer = forbidden_colors
    reverse_bfs_orders = first_visit_to_tree
    tree_set = TreeSet(g, forest, buffer, reverse_bfs_orders, ne)
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
    color_x::Integer,
    # modified
    first_visit_to_tree::AbstractVector{<:Tuple},
    forbidden_colors::AbstractVector{<:Integer},
    forest::Forest{<:Integer},
)
    root_wx = find_root!(forest, index_wx)  # root of the 2-colored tree T to which the edge wx belongs
    (p, q) = first_visit_to_tree[root_wx]
    if p != v  # T is being visited from vertex v for the first time
        first_visit_to_tree[root_wx] = (v, w)
    elseif q != w  # T is connected to vertex v via at least two edges
        forbidden_colors[color_x] = v
    end
    return nothing
end

function _grow_star!(
    # not modified
    v::Integer,
    w::Integer,
    index_vw::Integer,
    color_w::Integer,
    # modified
    first_neighbor::AbstractVector{<:Tuple},
    forest::Forest{<:Integer},
)
    # Create a new tree T_{vw} consisting only of edge vw
    (p, q, index_pq) = first_neighbor[color_w]
    if p != v  # a neighbor of v with color[w] encountered for the first time
        first_neighbor[color_w] = (v, w, index_vw)
    else  # merge T_{vw} with a two-colored star being grown around v
        root_vw = find_root!(forest, index_vw)
        root_pq = find_root!(forest, index_pq)
        root_union!(forest, root_vw, root_pq)
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
    root_vw = find_root!(forest, index_vw)
    root_wx = find_root!(forest, index_wx)
    if root_vw != root_wx
        root_union!(forest, root_vw, root_wx)
    end
    return nothing
end

"""
    TreeSet

Encode a set of 2-colored trees resulting from the [`acyclic_coloring`](@ref) algorithm.

# Fields

$TYPEDFIELDS
"""
struct TreeSet{T}
    reverse_bfs_orders::Vector{Tuple{T,T}}
    is_star::Vector{Bool}
    tree_edge_indices::Vector{T}
    nt::T
end

function TreeSet(
    g::AdjacencyGraph{T},
    forest::Forest{T},
    buffer::AbstractVector{T},
    reverse_bfs_orders::Vector{Tuple{T,T}},
    ne::Integer,
) where {T}
    S = pattern(g)
    edge_to_index = edge_indices(g)
    nv = nb_vertices(g)
    (; nt, ranks, parents) = forest

    # root_to_tree is a vector that maps a tree's root to the index of the tree
    # We can recycle the vector "ranks" because we don't need it anymore to merge trees
    root_to_tree = ranks
    fill!(root_to_tree, zero(T))

    # vector specifying the starting and ending indices of edges for each tree
    tree_edge_indices = zeros(T, nt + 1)

    # number of roots found
    nr = 0

    # determine the number of edges for each tree and map each root to a tree index
    for index_edge in 1:ne
        root = find_root!(forest, index_edge)

        # create a mapping between roots and tree indices
        if iszero(root_to_tree[root])
            nr += 1
            root_to_tree[root] = nr
        end

        # index of the tree that contains this edge
        index_tree = root_to_tree[root]

        # Update the number of edges for the current tree (shifted by 1 to facilitate the final cumsum)
        tree_edge_indices[index_tree + 1] += 1
    end

    # nvmax is the number of vertices in the largest tree of the forest
    # Note: the number of vertices in a tree is equal the number of edges plus one
    nvmax = maximum(tree_edge_indices) + one(T)

    # Vector containing the list of vertices, grouped by tree (each vertex appears once for every tree it belongs to)
    # Note: the total number of edges in the graph is "ne", so there are "ne + nt" vertices across all trees
    tree_vertices = Vector{T}(undef, ne + nt)

    # Provide the positions of the first and last neighbors for each vertex in "tree_vertices", within the tree to which the vertex belongs
    # These positions refer to indices in the vector "tree_neighbors"
    tree_neighbor_indices = zeros(T, ne + nt + 1)

    # Packed representation of the neighbors of each vertex in "tree_vertices"
    tree_neighbors = Vector{T}(undef, 2 * ne)

    # Track the positions for inserting vertices and neighbors per tree
    vertex_position = Vector{T}(undef, nt)
    neighbor_position = Vector{T}(undef, nt)

    # Compute starting positions for vertices and neighbors in each tree
    if nt > 0
        vertex_position[1] = zero(T)
        neighbor_position[1] = zero(T)
    end
    for k in 2:nt
        # Note: tree_edge_indices[k] is the number of edges in the tree k-1
        vertex_position[k] = vertex_position[k - 1] + tree_edge_indices[k] + 1
        neighbor_position[k] = neighbor_position[k - 1] + 2 * tree_edge_indices[k]
    end

    # Record the most recent vertex from which each tree is visited
    visited_trees = zeros(T, nt)

    rvS = rowvals(S)
    for j in axes(S, 2)
        for pos in nzrange(S, j)
            i = rvS[pos]
            if i != j
                index_ij = edge_to_index[pos]

                # No need to call "find_root!" because paths have already been compressed
                root = parents[index_ij]

                # Index of the tree containing edge (i, j)
                index_tree = root_to_tree[root]

                # Position in tree_vertices where vertex j should be found or inserted
                vertex_index = vertex_position[index_tree]

                if visited_trees[index_tree] != j
                    # Mark the current tree as visited from vertex j
                    visited_trees[index_tree] = j

                    # Insert j into tree_vertices
                    vertex_position[index_tree] += 1
                    vertex_index += 1
                    tree_vertices[vertex_index] = j
                end

                # Append neighbor i to the list of neighbors of j in the tree
                neighbor_position[index_tree] += 1
                neighbor_index = neighbor_position[index_tree]
                tree_neighbors[neighbor_index] = i

                # Increment neighbor count for j in the tree (shifted by 1 to facilitate the final cumsum)
                tree_neighbor_indices[vertex_index + 1] += 1
            end
        end
    end

    # Compute a shifted cumulative sum of tree_edge_indices, starting from one
    tree_edge_indices[1] = one(T)
    for k in 2:(nt + 1)
        tree_edge_indices[k] += tree_edge_indices[k - 1]
    end

    # Compute a shifted cumulative sum of tree_neighbor_indices, starting from one
    tree_neighbor_indices[1] = 1
    for k in 2:(ne + nt + 1)
        tree_neighbor_indices[k] += tree_neighbor_indices[k - 1]
    end

    # degrees is a vector of integers that stores the degree of each vertex in a tree
    degrees = buffer

    # For each vertex in the current tree, reverse_mapping will hold its corresponding index in tree_vertices
    reverse_mapping = Vector{T}(undef, nv)

    # Create a queue with a fixed size nvmax
    queue = Vector{T}(undef, nvmax)

    # Determine if each tree in the forest is a star
    # In a star, at most one vertex has a degree strictly greater than one
    is_star = Vector{Bool}(undef, nt)

    # Number of edges treated
    num_edges_treated = zero(T)

    # reverse_bfs_orders contains the reverse breadth first (BFS) traversal order for each tree in the forest
    for k in 1:nt
        # Initialize the queue to store the leaves
        queue_start = 1
        queue_end = 0

        # Positions of the first and last vertices in the current tree
        # Note: tree_edge_indices contains the positions of the first and last edges,
        # so we add to add an offset k-1 between edge indices and vertex indices
        first_vertex = tree_edge_indices[k] + (k - 1)
        last_vertex = tree_edge_indices[k + 1] + (k - 1)

        # compute the degree of each vertex in the tree
        for index_vertex in first_vertex:last_vertex
            vertex = tree_vertices[index_vertex]
            degree =
                tree_neighbor_indices[index_vertex + 1] -
                tree_neighbor_indices[index_vertex]
            degrees[vertex] = degree

            # store a reverse mapping to get the position of the vertex in tree_vertices
            reverse_mapping[vertex] = index_vertex

            # the vertex is a leaf
            if degree == 1
                queue_end += 1
                queue[queue_end] = vertex
            end
        end

        # number of vertices in the tree
        nv_tree = tree_edge_indices[k + 1] - tree_edge_indices[k] + 1

        # Check that no more than one vertex has a degree strictly greater than one
        # "queue_end" currently represents the number of vertices considered as leaves in the tree before any pruning
        is_star[k] = queue_end >= nv_tree - 1

        # continue until all leaves are treated
        while queue_start <= queue_end
            leaf = queue[queue_start]
            queue_start += 1

            # Mark the vertex as removed
            degrees[leaf] = 0

            # Position of the leaf in tree_vertices
            index_leaf = reverse_mapping[leaf]

            # Positions of the first and last neighbors of the leaf in the current tree
            first_neighbor = tree_neighbor_indices[index_leaf]
            last_neighbor = tree_neighbor_indices[index_leaf + 1] - 1

            # Iterate over all neighbors of the leaf to be pruned
            for index_neighbor in first_neighbor:last_neighbor
                neighbor = tree_neighbors[index_neighbor]

                # Check if neighbor is the parent of the leaf or if it was a child before the tree was pruned
                if degrees[neighbor] != 0
                    # (leaf, neighbor) represents the next edge to visit during decompression
                    num_edges_treated += 1
                    reverse_bfs_orders[num_edges_treated] = (leaf, neighbor)

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

    return TreeSet(reverse_bfs_orders, is_star, tree_edge_indices, nt)
end
