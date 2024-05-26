"""
    check_structurally_orthogonal_columns(
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
function check_structurally_orthogonal_columns(
    A::AbstractMatrix, color::AbstractVector{<:Integer}; verbose::Bool=false
)
    if length(color) != size(A, 2)
        if verbose
            @warn "$(length(color)) colors provided for $(size(A, 2)) columns"
        end
        return false
    end
    group = color_groups(color)
    for (c, g) in enumerate(group)
        Ag = view(A, :, g)
        nonzeros_per_row = dropdims(count(!iszero, Ag; dims=2); dims=2)
        max_nonzeros_per_row, i = findmax(nonzeros_per_row)
        if max_nonzeros_per_row > 1
            if verbose
                incompatible_columns = g[findall(!iszero, view(Ag, i, :))]
                @warn "In color $c, columns $incompatible_columns all have nonzeros in row $i"
            end
            return false
        end
    end
    return true
end

"""
    check_symmetrically_orthogonal_columns(
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
function check_symmetrically_orthogonal_columns(
    A::AbstractMatrix, color::AbstractVector{<:Integer}; verbose::Bool=false
)
    checksquare(A)
    if length(color) != size(A, 2)
        if verbose
            @warn "$(length(color)) colors provided for $(size(A, 2)) columns"
        end
        return false
    end
    issymmetric(A) || return false
    group = color_groups(color)
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
                - in color ci=$ci, columns $gi_incompatible_columns all have nonzeros in row j=$j
                - in color cj=$cj, columns $gj_incompatible_columns all have nonzeros in row i=$i
                """
            end
            return false
        end
    end
    return true
end
