## Postprocessing

function postprocess!(
    color::AbstractVector{<:Integer},
    star_or_tree_set::Union{StarSet,TreeSet},
    g::AdjacencyGraph,
    offsets::AbstractVector{<:Integer},
    postprocessing_minimizes::Symbol,
)
    # flag which colors are actually used during decompression
    nb_colors = maximum(color)
    color_used = zeros(Bool, nb_colors)

    # nonzero diagonal coefficients force the use of their respective color (there can be no neutral colors if the diagonal is fully nonzero)
    if !augmented_graph(g)
        S = pattern(g)
        for i in axes(S, 1)
            if !iszero(S[i, i])
                color_used[color[i]] = true
            end
        end

        if star_or_tree_set isa StarSet
            # star_or_tree_set is a StarSet
            postprocess_star_coloring!(g, color_used, color, star_or_tree_set, offsets)
        else
            # star_or_tree_set is a TreeSet
            postprocess_acyclic_coloring!(color_used, color, star_or_tree_set, offsets)
        end
    else
        row_color_used = zeros(Bool, nb_colors)
        column_color_used = color_used

        if star_or_tree_set isa StarSet
            # star_or_tree_set is a StarSet
            postprocess_star_bicoloring!(g, row_color_used, column_color_used, color, star_or_tree_set, offsets, postprocessing_minimizes)
        else
            # star_or_tree_set is a TreeSet
            postprocess_acyclic_bicoloring!(row_color_used, column_color_used, color, star_or_tree_set, offsets, postprocessing_minimizes)
        end

        # Identify colors that are used in either the row or column partition
        # color_used = row_color_used .| column_color_used
        color_used .|= row_color_used
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

function postprocess_star_coloring!(
    g::AdjacencyGraph,
    color_used::Vector{Bool},
    color::AbstractVector{<:Integer},
    star_set::StarSet,
    occurrences::AbstractVector{<:Integer},
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
        fill!(occurrences, 0)
        all_trivial_stars_treated = true

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
                        if color_used[color[h]]
                            # The current hub of this trivial star is already a hub in a non-trivial star
                            hub[s] = h
                        else
                            spoke = h == j ? i : j
                            if color_used[color[spoke]]
                                # The current spoke of this trivial star is also a hub in a non-trivial star
                                # Switch the hub and the spoke to avoid adding one more used color
                                hub[s] = spoke
                            else
                                all_trivial_stars_treated = false
                                # Increment the occurrence count of vertices i and j within the remaining set of trivial stars
                                occurrences[i] += 1
                                occurrences[j] += 1
                            end
                        end
                    end
                end
            end
        end

        # Only trivial stars, where both vertices can be promoted as hubs, remain.
        if !all_trivial_stars_treated
            rvS = rowvals(S)
            for j in axes(S, 2)
                for k in nzrange(S, j)
                    i = rvS[k]
                    if i > j
                        index_ij = edge_to_index[k]
                        s = star[index_ij]
                        h = hub[s]
                        # The hub of this trivial star is still unknown
                        if h < 0
                            # We need to decide who is the hub
                            if !color_used[color[i]] && !color_used[color[j]]
                                # We use the vertex with the highest occurrence as the hub
                                # This is a heuristic to maximize the number of vertices with a neutral color
                                # and may indirectly reduce the number of colors needed
                                if occurrences[j] > occurrences[i]
                                    hub[s] = j
                                    color_used[color[j]] = true
                                else
                                    hub[s] = i
                                    color_used[color[i]] = true
                                end
                            else
                                # Previously processed trivial stars determined the hub vertex for this star
                                if color_used[color[i]]
                                    hub[s] = i
                                else
                                    hub[s] = j
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return color_used
end

function postprocess_star_bicoloring!(
    g::AdjacencyGraph,
    row_color_used::Vector{Bool},
    column_color_used::Vector{Bool},
    color::AbstractVector{<:Integer},
    star_set::StarSet,
    occurrences::AbstractVector{<:Integer},
    postprocessing_minimizes::Symbol=:all_colors,
)
    S = pattern(g)
    edge_to_index = edge_indices(g)

    # only the colors of the hubs are used
    (; star, hub) = star_set
    nb_trivial_stars = 0

    # size of the original matrix on which we want to perform bicoloring
    (m, n) = g.original_size

    # Iterate through all non-trivial stars
    for s in eachindex(hub)
        h = hub[s]
        if h > 0
            if h ≤ n
                column_color_used[color[h]] = true
            else
                row_color_used[color[h]] = true
            end
        else
            nb_trivial_stars += 1
        end
    end

    # Process the trivial stars (if any)
    if nb_trivial_stars > 0
        fill!(occurrences, 0)
        all_trivial_stars_treated = true

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
                        if (h ≤ n ? column_color_used[color[h]] : row_color_used[color[h]])
                            # The current hub of this trivial star is already a hub in a non-trivial star
                            hub[s] = h
                        else
                            spoke = h == j ? i : j
                            if (spoke ≤ n ? column_color_used[color[spoke]] : row_color_used[color[spoke]])
                                # The current spoke of this trivial star is also a hub in a non-trivial star
                                # Switch the hub and the spoke to avoid adding one more used color
                                hub[s] = spoke
                            else
                                all_trivial_stars_treated = false
                                # Increment the occurrence count of vertices i and j within the remaining set of trivial stars
                                occurrences[i] += 1
                                occurrences[j] += 1
                            end
                        end
                    end
                end
            end
        end

        # Only trivial stars, where both vertices can be promoted as hubs, remain.
        # In the context of bicoloring, if we aim to minimize either the number of row colors or the number of column colors,
        # we can achieve optimal post-processing by choosing as hubs the vertices from the opposite partition.
        # This is optimal because we never increase the number of colors in the target partition during this phase,
        # and all preceding steps of the post-processing are deterministic.
        if !all_trivial_stars_treated
            rvS = rowvals(S)
            for j in axes(S, 2)
                for k in nzrange(S, j)
                    i = rvS[k]
                    if i > j
                        index_ij = edge_to_index[k]
                        s = star[index_ij]
                        h = hub[s]
                        # The hub of this trivial star is still unknown
                        if h < 0
                            # We need to decide who is the hub
                            if !row_color_used[color[i]] && !column_color_used[color[j]]
                                if postprocessing_minimizes == :row_colors
                                    # j belongs to a column partition in the context of bicoloring
                                    hub[s] = j
                                    column_color_used[color[j]] = true
                                elseif postprocessing_minimizes == :column_colors
                                    # i belongs to a row partition in the context of bicoloring
                                    hub[s] = i
                                    row_color_used[color[i]] = true
                                elseif postprocessing_minimizes == :all_colors
                                    # We use the vertex with the highest occurrence as the hub
                                    # This is a heuristic to maximize the number of vertices with a neutral color
                                    # and may indirectly reduce the number of colors needed
                                    if occurrences[j] > occurrences[i]
                                        hub[s] = j
                                        column_color_used[color[j]] = true
                                    else
                                        hub[s] = i
                                        row_color_used[color[i]] = true
                                    end
                                else
                                    error("The value postprocessing_minimizes = :$postprocessing_minimizes is not supported.")
                                end
                            else
                                # Previously processed trivial stars determined the hub vertex for this star
                                if row_color_used[color[i]]
                                    hub[s] = i
                                else
                                    hub[s] = j
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return row_color_used, column_color_used
end

function postprocess_acyclic_coloring!(
    color_used::Vector{Bool},
    color::AbstractVector{<:Integer},
    tree_set::TreeSet,
    occurrences::AbstractVector{<:Integer},
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
        fill!(occurrences, 0)
        all_trivial_trees_treated = true

        for k in 1:nt
            # Position of the first edge in the tree
            first = tree_edge_indices[k]

            # Total number of edges in the tree
            ne_tree = tree_edge_indices[k + 1] - first

            # Check if we have exactly one edge in the tree
            if ne_tree == 1
                (i, j) = reverse_bfs_orders[first]
                if color_used[color[j]]
                    # The current root of this trivial tree is already an internal node in a non-trivial tree
                else
                    if color_used[color[i]]
                        # The current leaf of this trivial tree is also an internal node in a non-trivial tree
                        # Switch the root and the leaf to avoid adding one more used color
                        reverse_bfs_orders[first] = (j, i)
                    else
                        all_trivial_trees_treated = false
                        # Increment the occurrence count of vertices i and j within the remaining set of trivial trees
                        occurrences[i] += 1
                        occurrences[j] += 1
                    end
                end
            end
        end

        # Only trivial trees, where both vertices can be promoted as roots, remain.
        if !all_trivial_trees_treated
           for k in 1:nt
                # Position of the first edge in the tree
                first = tree_edge_indices[k]

                # Total number of edges in the tree
                ne_tree = tree_edge_indices[k + 1] - first

                # Check if we have exactly one edge in the tree
                if ne_tree == 1
                    (i, j) = reverse_bfs_orders[first]
                    if !color_used[color[i]] && !color_used[color[j]]
                        # We use the vertex with the highest occurrence as the root
                        # This is a heuristic to maximize the number of vertices with a neutral color
                        # and may indirectly reduce the number of colors needed
                        if occurrences[j] > occurrences[i]
                            color_used[color[j]] = true
                        else
                            reverse_bfs_orders[first] = (j, i)
                            color_used[color[i]] = true
                        end
                    else
                        # Previously processed trivial trees determined the root vertex for this tree
                        # Ensure that the root vertex has a used color for decompression
                        if color_used[color[i]] && !color_used[color[j]]
                            reverse_bfs_orders[first] = (j, i)
                        end
                    end
                end
            end
        end
    end
    return color_used
end

function postprocess_acyclic_bicoloring!(
    row_color_used::Vector{Bool},
    column_color_used::Vector{Bool},
    color::AbstractVector{<:Integer},
    tree_set::TreeSet,
    occurrences::AbstractVector{<:Integer},
    postprocessing_minimizes::Symbol=:all_colors,
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
                (leaf, hub) = reverse_bfs_orders[first]
                if hub ≤ leaf
                    column_color_used[color[hub]] = true
                else
                    row_color_used[color[hub]] = true
                end
            else
                # It is not a star and both colors are needed during the decompression
                (i, j) = reverse_bfs_orders[first]
                if i < j
                    column_color_used[color[i]] = true
                    row_color_used[color[j]] = true
                else
                    row_color_used[color[i]] = true
                    column_color_used[color[j]] = true
                end
            end
        else
            nb_trivial_trees += 1
        end
    end

    # Process the trivial trees (if any)
    if nb_trivial_trees > 0
        fill!(occurrences, 0)
        all_trivial_trees_treated = true

        for k in 1:nt
            # Position of the first edge in the tree
            first = tree_edge_indices[k]

            # Total number of edges in the tree
            ne_tree = tree_edge_indices[k + 1] - first

            # Check if we have exactly one edge in the tree
            if ne_tree == 1
                (i, j) = reverse_bfs_orders[first]
                if (i < j ? row_color_used[color[j]] : column_color_used[color[j]])
                    # The current root of this trivial tree is already an internal node in a non-trivial tree
                else
                    if (i < j ? column_color_used[color[i]] : row_color_used[color[i]])
                        # The current leaf of this trivial tree is also an internal node in a non-trivial tree
                        # Switch the root and the leaf to avoid adding one more used color
                        reverse_bfs_orders[first] = (j, i)
                    else
                        all_trivial_trees_treated = false
                        # Increment the occurrence count of vertices i and j within the remaining set of trivial trees
                        occurrences[i] += 1
                        occurrences[j] += 1
                    end
                end
            end
        end

        # Only trivial trees, where both vertices can be promoted as roots, remain.
        # In the context of bicoloring, if we aim to minimize either the number of row colors or the number of column colors,
        # we can achieve optimal post-processing by choosing as roots the vertices from the opposite partition.
        # This is optimal because we never increase the number of colors in the target partition during this phase,
        # and all preceding steps of the post-processing are deterministic.
        if !all_trivial_trees_treated
           for k in 1:nt
                # Position of the first edge in the tree
                first = tree_edge_indices[k]

                # Total number of edges in the tree
                ne_tree = tree_edge_indices[k + 1] - first

                # Check if we have exactly one edge in the tree
                if ne_tree == 1
                    (i, j) = reverse_bfs_orders[first]
                    if (i < j ? !column_color_used[color[i]] && !row_color_used[color[j]] : !row_color_used[color[i]] && !column_color_used[color[j]])
                        if postprocessing_minimizes == :row_colors
                            # v belongs to a column partition in the context of bicoloring
                            v = min(i,j)
                            column_color_used[color[v]] = true
                            if v == i
                                reverse_bfs_orders[first] = (j, i)
                            end
                        elseif postprocessing_minimizes == :column_colors
                            # v belongs to a row partition in the context of bicoloring
                            v = max(i,j)
                            row_color_used[color[v]] = true
                            if v == i
                                reverse_bfs_orders[first] = (j, i)
                            end
                        elseif postprocessing_minimizes == :all_colors
                            # We use the vertex with the highest occurrence as the root
                            # This is a heuristic to maximize the number of vertices with a neutral color
                            # and may indirectly reduce the number of colors needed
                            if occurrences[j] > occurrences[i]
                                if i < j
                                    row_color_used[color[j]] = true
                                else
                                    column_color_used[color[j]] = true
                                end
                            else
                                reverse_bfs_orders[first] = (j, i)
                                if i < j
                                    column_color_used[color[i]] = true
                                else
                                    row_color_used[color[i]] = true
                                end
                            end
                        else
                            error("The value postprocessing_minimizes = :$postprocessing_minimizes is not supported.")
                        end
                    else
                        # Previously processed trivial trees determined the root vertex for this tree
                        # Ensure that the root vertex has a used color for decompression
                        if (i < j ? column_color_used[color[i]] && !row_color_used[color[j]] : row_color_used[color[i]] && !column_color_used[color[j]])
                            reverse_bfs_orders[first] = (j, i)
                        end
                    end
                end
            end
        end
    end
    return row_color_used, column_color_used
end
