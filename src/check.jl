function proper_length_coloring(
    A::AbstractMatrix, color::AbstractVector{<:Integer}; verbose::Bool
)
    if length(color) != size(A, 2)
        if verbose
            @warn "$(length(color)) colors provided for $(size(A, 2)) columns."
        end
        return false
    end
    return true
end

"""
    structurally_orthogonal_columns(
        A::AbstractMatrix, color::AbstractVector{<:Integer}
        verbose=false
    )

Return `true` if coloring the columns of the matrix `A` with the vector `color` results in a partition that is structurally orthogonal, and `false` otherwise.
    
A partition of the columns of a matrix `A` is _structurally orthogonal_ if, for every nonzero element `A[i, j]`, the group containing column `A[:, j]` has no other column with a nonzero in row `i`.

!!! warning
    This function is not coded with efficiency in mind, it is designed for small-scale tests.

# References

> [_What Color Is Your Jacobian? Graph Coloring for Computing Derivatives_](https://epubs.siam.org/doi/10.1137/S0036144504444711), Gebremedhin et al. (2005)
"""
function structurally_orthogonal_columns(
    A::AbstractMatrix, color::AbstractVector{<:Integer}; verbose::Bool=false
)
    if !proper_length_coloring(A, color; verbose)
        return false
    end
    group = group_by_color(color)
    for (c, g) in enumerate(group)
        Ag = view(A, :, g)
        nonzeros_per_row = only(eachcol(count(!iszero, Ag; dims=2)))
        max_nonzeros_per_row, i = findmax(nonzeros_per_row)
        if max_nonzeros_per_row > 1
            if verbose
                incompatible_columns = g[findall(!iszero, view(Ag, i, :))]
                @warn "In color $c, columns $incompatible_columns all have nonzeros in row $i."
            end
            return false
        end
    end
    return true
end

"""
    symmetrically_orthogonal_columns(
        A::AbstractMatrix, color::AbstractVector{<:Integer};
        verbose=false
    )

Return `true` if coloring the columns of the symmetric matrix `A` with the vector `color` results in a partition that is symmetrically orthogonal, and `false` otherwise.
    
A partition of the columns of a symmetrix matrix `A` is _symmetrically orthogonal_ if, for every nonzero element `A[i, j]`, either of the following statements holds:

1. the group containing the column `A[:, j]` has no other column with a nonzero in row `i`
2. the group containing the column `A[:, i]` has no other column with a nonzero in row `j`

!!! warning
    This function is not coded with efficiency in mind, it is designed for small-scale tests.

# References

> [_What Color Is Your Jacobian? Graph Coloring for Computing Derivatives_](https://epubs.siam.org/doi/10.1137/S0036144504444711), Gebremedhin et al. (2005)
"""
function symmetrically_orthogonal_columns(
    A::AbstractMatrix, color::AbstractVector{<:Integer}; verbose::Bool=false
)
    checksquare(A)
    if !proper_length_coloring(A, color; verbose)
        return false
    end
    issymmetric(A) || return false
    group = group_by_color(color)
    for i in axes(A, 2), j in axes(A, 2)
        iszero(A[i, j]) && continue
        ci, cj = color[i], color[j]
        gi, gj = group[ci], group[cj]
        A_gj_rowi = view(A, i, gj)
        A_gi_rowj = view(A, j, gi)
        nonzeros_gj_rowi = count(!iszero, A_gj_rowi)
        nonzeros_gi_rowj = count(!iszero, A_gi_rowj)
        if nonzeros_gj_rowi > 1 && nonzeros_gi_rowj > 1
            if verbose
                gj_incompatible_columns = gj[findall(!iszero, A_gj_rowi)]
                gi_incompatible_columns = gi[findall(!iszero, A_gi_rowj)]
                @warn """
                For coefficient (i=$i, j=$j) with column colors (ci=$ci, cj=$cj):
                - In color ci=$ci, columns $gi_incompatible_columns all have nonzeros in row j=$j.
                - In color cj=$cj, columns $gj_incompatible_columns all have nonzeros in row i=$i.
                """
            end
            return false
        end
    end
    return true
end

"""
    directly_recoverable_columns(
        A::AbstractMatrix, color::AbstractVector{<:Integer}
        verbose=false
    )

Return `true` if coloring the columns of the symmetric matrix `A` with the vector `color` results in a column-compressed representation that preserves every unique value, thus making direct recovery possible.

!!! warning
    This function is not coded with efficiency in mind, it is designed for small-scale tests.

# References

> [_What Color Is Your Jacobian? Graph Coloring for Computing Derivatives_](https://epubs.siam.org/doi/10.1137/S0036144504444711), Gebremedhin et al. (2005)
"""
function directly_recoverable_columns(
    A::AbstractMatrix, color::AbstractVector{<:Integer}; verbose::Bool=false
)
    if !proper_length_coloring(A, color; verbose)
        return false
    end
    group = group_by_color(color)
    B = stack(group; dims=2) do g
        dropdims(sum(A[:, g]; dims=2); dims=2)
    end
    A_unique = Set(unique(A))
    B_unique = Set(unique(B))
    if !issubset(A_unique, B_unique)
        if verbose
            @warn "Coefficients $(sort(collect(setdiff(A_unique, B_unique)))) are not directly recoverable."
            return false
        end
        return false
    end
    return true
end

"""
    valid_dynamic_order(g::AdjacencyGraph, π::AbstractVector{Int}, order::DynamicDegreeBasedOrder)
    valid_dynamic_order(bg::AdjacencyGraph, ::Val{side}, π::AbstractVector{Int}, order::DynamicDegreeBasedOrder)

Check that a permutation `π` corresponds to a valid application of a [`DynamicDegreeBasedOrder`](@ref).

This is done by checking, for each ordered vertex, that its back- or forward-degree was the smallest or largest among the remaining vertices (the specifics depend on the order parameters).

!!! warning
    This function is not coded with efficiency in mind, it is designed for small-scale tests.
"""
function valid_dynamic_order(
    g::AdjacencyGraph, π::AbstractVector{Int}, ::DynamicDegreeBasedOrder{degtype,direction}
) where {degtype,direction}
    length(π) != nb_vertices(g) && return false
    length(unique(π)) != nb_vertices(g) && return false
    for i in eachindex(π)
        vi = π[i]
        yet_to_be_ordered = direction == :low2high ? π[i:end] : π[begin:i]
        considered_for_degree = degtype == :back ? π[begin:(i - 1)] : π[(i + 1):end]
        di = degree_in_subset(g, vi, considered_for_degree)
        considered_for_degree_switched = copy(considered_for_degree)
        for vj in yet_to_be_ordered
            replace!(considered_for_degree_switched, vj => vi)
            dj = degree_in_subset(g, vj, considered_for_degree_switched)
            replace!(considered_for_degree_switched, vi => vj)
            if direction == :low2high
                dj > di && return false
            else
                dj < di && return false
            end
        end
    end
    return true
end

function valid_dynamic_order(
    g::BipartiteGraph,
    ::Val{side},
    π::AbstractVector{Int},
    ::DynamicDegreeBasedOrder{degtype,direction},
) where {side,degtype,direction}
    length(π) != nb_vertices(g, Val(side)) && return false
    length(unique(π)) != nb_vertices(g, Val(side)) && return false
    for i in eachindex(π)
        vi = π[i]
        yet_to_be_ordered = direction == :low2high ? π[i:end] : π[begin:i]
        considered_for_degree = degtype == :back ? π[begin:(i - 1)] : π[(i + 1):end]
        di = degree_dist2_in_subset(g, Val(side), vi, considered_for_degree)
        considered_for_degree_switched = copy(considered_for_degree)
        for vj in yet_to_be_ordered
            replace!(considered_for_degree_switched, vj => vi)
            dj = degree_dist2_in_subset(g, Val(side), vj, considered_for_degree_switched)
            replace!(considered_for_degree_switched, vi => vj)
            if direction == :low2high
                dj > di && return false
            else
                dj < di && return false
            end
        end
    end
    return true
end
