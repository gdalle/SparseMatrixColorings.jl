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
    bicoloring = augmented_graph(g)
    if !bicoloring
        S = pattern(g)
        for i in axes(S, 1)
            if !iszero(S[i, i])
                color_used[color[i]] = true
            end
        end
    end

    # When bicoloring is false, column_color_used and row_color_used point to the same memory
    row_color_used = bicoloring ? zeros(Bool, nb_colors) : color_used
    column_color_used = color_used

    if star_or_tree_set isa StarSet
        # star_or_tree_set is a StarSet
        postprocess_with_star_set!(
            bicoloring,
            g,
            row_color_used,
            column_color_used,
            color,
            star_or_tree_set,
            postprocessing_minimizes,
        )
    else
        # star_or_tree_set is a TreeSet
        postprocess_with_tree_set!(
            bicoloring,
            row_color_used,
            column_color_used,
            color,
            star_or_tree_set,
            postprocessing_minimizes,
        )
    end

    # if at least one of the colors is not used, modify the color assignments of vertices
    has_neutral_color = if bicoloring
        any(!, row_color_used) || any(!, column_color_used)
    else
        any(!, color_used)
    end

    if has_neutral_color
        # size of the original matrix on which we want to perform coloring or bicoloring
        (m, n) = g.original_size

        # count the number of unused colors
        num_unused_colors = 0

        # count how many color indices are skipped before each color,
        # in order to compact the color indexing after removing unused colors
        for ci in 1:nb_colors
            ci_required =
                bicoloring ? row_color_used[ci] || column_color_used[ci] : color_used[ci]
            if ci_required
                offsets[ci] = num_unused_colors
            else
                num_unused_colors += 1
            end
        end

        # replace unused colors by the neutral color and compact the remaining color indices
        for i in eachindex(color)
            ci = color[i]
            ci_used = (i ≤ n) ? column_color_used[ci] : row_color_used[ci]
            if !ci_used
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
    bicoloring::Bool,
    g::AdjacencyGraph,
    row_color_used::Vector{Bool},
    column_color_used::Vector{Bool},
    color::AbstractVector{<:Integer},
    star_set::StarSet,
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
        # When bicoloring is false, row_color_counts and column_color_counts point to the same memory
        nv = length(color)
        nb_colors = length(row_color_used)
        visited_vertices = zeros(Bool, nv)
        row_color_counts = zeros(Int, nb_colors)
        column_color_counts = bicoloring ? zeros(Int, nb_colors) : row_color_counts
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
                        if row_color_used[color[i]]
                            # The vertex i is already a hub in a non-trivial star
                            hub[s] = i
                        else
                            if column_color_used[color[j]]
                                # The vertex j is already a hub in a non-trivial star
                                hub[s] = j
                            else
                                all_trivial_stars_treated = false
                                # Count how many vertices of each color appear among the remaining trivial stars.
                                # Each vertex is counted at most once, using `visited_vertices` to avoid duplicates
                                # when a vertex belongs to multiple trivial stars.
                                if !visited_vertices[i]
                                    visited_vertices[i] = true
                                    row_color_counts[color[i]] += 1
                                end
                                if !visited_vertices[j]
                                    visited_vertices[j] = true
                                    column_color_counts[color[j]] += 1
                                end
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
                                if !bicoloring || postprocessing_minimizes == :all_colors
                                    # Choose as hub the vertex whose color is most frequent among the trivial stars.
                                    # Colors with smaller `color_counts` are easier to keep unused
                                    # if their vertices remain spokes instead of hubs.
                                    # This is an heuristic to try to reduce the number of colors used.
                                    #
                                    # In case of a tie, we prefer to preserve colors in the row partition.
                                    #
                                    # Note that this heuristic also depends on the order in which
                                    # the trivial stars are processed, especially when there are ties in `color_counts`.
                                    if row_color_counts[color[i]] >
                                        column_color_counts[color[j]]
                                        row_color_used[color[i]] = true
                                        hub[s] = i
                                    else
                                        column_color_used[color[j]] = true
                                        hub[s] = j
                                    end
                                elseif postprocessing_minimizes == :row_colors
                                    # j belongs to a column partition in the context of bicoloring
                                    hub[s] = j
                                    column_color_used[color[j]] = true
                                elseif postprocessing_minimizes == :column_colors
                                    # i belongs to a row partition in the context of bicoloring
                                    hub[s] = i
                                    row_color_used[color[i]] = true
                                else
                                    error(
                                        "The value postprocessing_minimizes = :$postprocessing_minimizes is not supported.",
                                    )
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
    return nothing
end

function postprocess_with_tree_set!(
    bicoloring::Bool,
    row_color_used::Vector{Bool},
    column_color_used::Vector{Bool},
    color::AbstractVector{<:Integer},
    tree_set::TreeSet,
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
                if hub < leaf
                    column_color_used[color[hub]] = true
                else
                    row_color_used[color[hub]] = true
                end
            else
                # It is not a star and both colors are needed during the decompression
                (i, j) = reverse_bfs_orders[first]
                v_col = min(i, j)
                v_row = max(i, j)
                row_color_used[color[v_row]] = true
                column_color_used[color[v_col]] = true
            end
        else
            nb_trivial_trees += 1
        end
    end

    # Process the trivial trees (if any)
    if nb_trivial_trees > 0
        # When bicoloring is false, row_color_counts and column_color_counts point to the same memory
        nv = length(color)
        nb_colors = length(row_color_used)
        visited_vertices = zeros(Bool, nv)
        row_color_counts = zeros(Int, nb_colors)
        column_color_counts = bicoloring ? zeros(Int, nb_colors) : row_color_counts
        all_trivial_trees_treated = true

        for k in 1:nt
            # Position of the first edge in the tree
            first = tree_edge_indices[k]

            # Total number of edges in the tree
            ne_tree = tree_edge_indices[k + 1] - first

            # Check if we have exactly one edge in the tree
            if ne_tree == 1
                (i, j) = reverse_bfs_orders[first]
                v_col = min(i, j)
                v_row = max(i, j)
                if column_color_used[color[v_col]]
                    # The vertex v_col is already an internal node in a non-trivial tree
                    reverse_bfs_orders[first] = (v_row, v_col)
                else
                    if row_color_used[color[v_row]]
                        # The vertex v_row is already an internal node in a non-trivial tree
                        reverse_bfs_orders[first] = (v_col, v_row)
                    else
                        all_trivial_trees_treated = false
                        # Count how many vertices of each color appear among the remaining trivial trees.
                        # Each vertex is counted at most once, using `visited_vertices` to avoid duplicates
                        # when a vertex belongs to multiple trivial trees.
                        if !visited_vertices[v_row]
                            visited_vertices[v_row] = true
                            row_color_counts[color[v_row]] += 1
                        end
                        if !visited_vertices[v_col]
                            visited_vertices[v_col] = true
                            column_color_counts[color[v_col]] += 1
                        end
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
                    v_col = min(i, j)
                    v_row = max(i, j)
                    if !column_color_used[color[v_col]] && !row_color_used[color[v_row]]
                        if !bicoloring || postprocessing_minimizes == :all_colors
                            # Choose as root the vertex whose color is most frequent among the trivial trees.
                            # Colors with smaller `color_counts` are easier to keep unused
                            # if their vertices remain leaves instead of roots.
                            # This is an heuristic to try to reduce the number of colors used.
                            #
                            # In case of a tie, we prefer to preserve colors in the row partition.
                            #
                            # Note that this heuristic also depends on the order in which
                            # the trivial trees are processed, especially when there are ties in `color_counts`.
                            if row_color_counts[color[v_row]] >
                                column_color_counts[color[v_col]]
                                row_color_used[color[v_row]] = true
                                reverse_bfs_orders[first] = (v_col, v_row)
                            else
                                column_color_used[color[v_col]] = true
                                reverse_bfs_orders[first] = (v_row, v_col)
                            end
                        elseif postprocessing_minimizes == :row_colors
                            # v_col belongs to a column partition in the context of bicoloring
                            column_color_used[color[v_col]] = true
                            reverse_bfs_orders[first] = (v_row, v_col)
                        elseif postprocessing_minimizes == :column_colors
                            # v_row belongs to a row partition in the context of bicoloring
                            row_color_used[color[v_row]] = true
                            reverse_bfs_orders[first] = (v_col, v_row)
                        else
                            error(
                                "The value postprocessing_minimizes = :$postprocessing_minimizes is not supported.",
                            )
                        end
                    else
                        # Previously processed trivial trees determined the root vertex for this tree
                        # Ensure that the root vertex has a used color for decompression
                        if column_color_used[color[v_col]] && !row_color_used[color[v_row]]
                            reverse_bfs_orders[first] = (v_row, v_col)
                        end
                        if !column_color_used[color[v_col]] && row_color_used[color[v_row]]
                            reverse_bfs_orders[first] = (v_col, v_row)
                        end
                    end
                end
            end
        end
    end
    return nothing
end
