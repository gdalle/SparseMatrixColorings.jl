## Postprocessing

function postprocess!(
    color::AbstractVector{<:Integer},
    star_or_tree_set::Union{StarSet,TreeSet},
    g::AdjacencyGraph,
    offsets::AbstractVector{<:Integer},
)
    S = pattern(g)
    edge_to_index = edge_indices(g)
    # flag which colors are actually used during decompression
    nb_colors = maximum(color)
    color_used = zeros(Bool, nb_colors)

    # nonzero diagonal coefficients force the use of their respective color (there can be no neutral colors if the diagonal is fully nonzero)
    if !augmented_graph(g)
        for i in axes(S, 1)
            if !iszero(S[i, i])
                color_used[color[i]] = true
            end
        end
    end

    if star_or_tree_set isa StarSet
        # star_or_tree_set is a StarSet
        postprocess_with_star_set!(g, color_used, color, star_or_tree_set)
    else
        # star_or_tree_set is a TreeSet
        postprocess_with_tree_set!(color_used, color, star_or_tree_set)
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

function postprocess_with_star_set!(
    g::AdjacencyGraph,
    color_used::Vector{Bool},
    color::AbstractVector{<:Integer},
    star_set::StarSet,
)
    S = pattern(g)
    edge_to_index = edge_indices(g)

    # only the colors of the hubs are used
    (; star, hub) = star_set
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
    return color_used
end

function postprocess_with_tree_set!(
    color_used::Vector{Bool},
    color::AbstractVector{<:Integer},
    tree_set::TreeSet,
)
    # only the colors of non-leaf vertices are used
    (; reverse_bfs_orders, is_star, tree_edge_indices, nt) = tree_set
    nb_trivial_trees = 0

    # Iterate through all non-trivial trees
    for k in 1:nt
        # Position of the first edge in the tree
        first = tree_edge_indices[k]

        # Total number of edges in the tree
        ne_tree = tree_edge_indices[k + 1] - first

        # Check if we have more than one edge in the tree (non-trivial tree)
        if ne_tree > 1
            # Determine if the tree is a star
            if is_star[k]
                # It is a non-trivial star and only the color of the hub is needed
                (_, hub) = reverse_bfs_orders[first]
                color_used[color[hub]] = true
            else
                # It is not a star and both colors are needed during the decompression
                (i, j) = reverse_bfs_orders[first]
                color_used[color[i]] = true
                color_used[color[j]] = true
            end
        else
            nb_trivial_trees += 1
        end
    end

    # Process the trivial trees (if any)
    if nb_trivial_trees > 0
        for k in 1:nt
            # Position of the first edge in the tree
            first = tree_edge_indices[k]

            # Total number of edges in the tree
            ne_tree = tree_edge_indices[k + 1] - first

            # Check if we have exactly one edge in the tree
            if ne_tree == 1
                (i, j) = reverse_bfs_orders[first]
                if color_used[color[i]]
                    # Make i the root to avoid possibly adding one more used color
                    # Switch it with the (only) leaf
                    reverse_bfs_orders[first] = (j, i)
                else
                    # Keep j as the root
                    color_used[color[j]] = true
                end
            end
        end
    end
    return color_used
end
