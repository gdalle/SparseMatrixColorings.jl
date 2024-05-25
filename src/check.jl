"""
    check_structurally_orthogonal_columns(
        A::AbstractMatrix, colors::AbstractVector{<:Integer}
        verbose=false
    )

Return `true` if coloring the columns of the matrix `A` with the vector `colors` results in a partition that is structurally orthogonal, and `false` otherwise.
    
A partition of the columns of a matrix `A` is _structurally orthogonal_ if, for every nonzero element `A[i, j]`, the group containing column `A[:, j]` has no other column with a nonzero in row `i`.

!!! warning
    This function is not coded with efficiency in mind, it is designed for small-scale tests.
"""
function check_structurally_orthogonal_columns(
    A::AbstractMatrix, colors::AbstractVector{<:Integer}; verbose::Bool=true
)
    groups = color_groups(colors)
    for (c, g) in enumerate(groups)
        Ag = @view A[:, g]
        nonzeros_per_row = dropdims(count(!iszero, Ag; dims=2); dims=2)
        max_nonzeros_per_row, i = findmax(nonzeros_per_row)
        if max_nonzeros_per_row > 1
            verbose && @warn "Columns $g (with color $c) share nonzeros in row $i"
            return false
        end
    end
    return true
end

"""
    check_structurally_orthogonal_rows(
        A::AbstractMatrix, colors::AbstractVector{<:Integer};
        verbose=false
    )

Return `true` if coloring the rows of the matrix `A` with the vector `colors` results in a partition that is structurally orthogonal, and `false` otherwise.
    
A partition of the rows of a matrix `A` is _structurally orthogonal_ if, for every nonzero element `A[i, j]`, the group containing row `A[i, :]` has no other row with a nonzero in column `j`.

!!! warning
    This function is not coded with efficiency in mind, it is designed for small-scale tests.
"""
function check_structurally_orthogonal_rows(
    A::AbstractMatrix, colors::AbstractVector{<:Integer}; verbose::Bool=true
)
    groups = color_groups(colors)
    for (c, g) in enumerate(groups)
        Ag = @view A[g, :]
        nonzeros_per_col = dropdims(count(!iszero, Ag; dims=1); dims=1)
        max_nonzeros_per_col, j = findmax(nonzeros_per_col)
        if max_nonzeros_per_col > 1
            verbose && @warn "Rows $g (with color $c) share nonzeros in column $j"
            return false
        end
    end
    return true
end

"""
    check_symmetrically_orthogonal(
        A::AbstractMatrix, colors::AbstractVector{<:Integer};
        verbose=false
    )

Return `true` if coloring the columns of the symmetric matrix `A` with the vector `colors` results in a partition that is symmetrically orthogonal, and `false` otherwise.
    
A partition of the columns of a symmetrix matrix `A` is _symmetrically orthogonal_ if, for every nonzero element `A[i, j]`, either

1. the group containing the column `A[:, j]` has no other column with a nonzero in row `i`
2. the group containing the column `A[:, i]` has no other column with a nonzero in row `j`

!!! warning
    This function is not coded with efficiency in mind, it is designed for small-scale tests.
"""
function check_symmetrically_orthogonal(
    A::AbstractMatrix, colors::AbstractVector{<:Integer}; verbose::Bool=true
)
    checksquare(A)
    issymmetric(A) || return false
    groups = color_groups(colors)
    for i in axes(A, 2), j in axes(A, 2)
        iszero(A[i, j]) && continue
        ki, kj = colors[i], colors[j]
        gi, gj = groups[ki], groups[kj]
        A_gj_rowi = view(A, i, gj)
        A_gi_rowj = view(A, j, gi)
        nonzeros_gj_rowi = count(!iszero, A_gj_rowi)
        nonzeros_gi_rowj = count(!iszero, A_gi_rowj)
        if nonzeros_gj_rowi > 1 && nonzeros_gi_rowj > 1
            verbose && @warn """
            For coefficient $((i, j)):
            - columns $gj (with color $kj) share nonzeros in row $i
            - columns $gi (with color $ki) share nonzeros in row $j
            """
            return false
        end
    end
    return true
end
