"""
    check_structurally_orthogonal_columns(A, colors; verbose=false)

Return `true` if coloring the columns of the matrix `A` with the vector `colors` results in a partition that is structurally orthogonal, and `false` otherwise.
    
Def 3.2: A partition of the columns of a matrix `A` is _structurally orthogonal_ if, for every nonzero element `A[i, j]`, the group containing column `A[:, j]` has no other column with a nonzero in row `i`.

Thm 3.5: The function [`distance2_column_coloring`](@ref) applied to the [`BipartiteGraph`](@ref) of `A` should return a suitable coloring.
"""
function check_structurally_orthogonal_columns(
    A::AbstractMatrix, colors::AbstractVector{<:Integer}; verbose=false
)
    for c in unique(colors)
        js = filter(j -> colors[j] == c, axes(A, 2))
        Ajs = @view A[:, js]
        nonzeros_per_row = count(!iszero, Ajs; dims=2)
        if maximum(nonzeros_per_row) > 1
            verbose && @warn "Color $c has columns $js sharing nonzeros"
            return false
        end
    end
    return true
end

"""
    check_structurally_orthogonal_rows(A, colors; verbose=false)

Return `true` if coloring the rows of the matrix `A` with the vector `colors` results in a partition that is structurally orthogonal, and `false` otherwise.
    
Def 3.2: A partition of the rows of a matrix `A` is _structurally orthogonal_ if, for every nonzero element `A[i, j]`, the group containing row `A[i, :]` has no other row with a nonzero in column `j`.

Thm 3.5: The function [`distance2_row_coloring`](@ref) applied to the [`BipartiteGraph`](@ref) of `A` should return a suitable coloring.
"""
function check_structurally_orthogonal_rows(
    A::AbstractMatrix, colors::AbstractVector{<:Integer}; verbose=false
)
    for c in unique(colors)
        is = filter(i -> colors[i] == c, axes(A, 1))
        Ais = @view A[is, :]
        nonzeros_per_column = count(!iszero, Ais; dims=1)
        if maximum(nonzeros_per_column) > 1
            verbose && @warn "Color $c has rows $is sharing nonzeros"
            return false
        end
    end
    return true
end

"""
    check_symmetrically_orthogonal(A, colors; verbose=false)

Return `true` if coloring the columns of the symmetric matrix `A` with the vector `colors` results in a partition that is symmetrically orthogonal, and `false` otherwise.
    
Def 4.2: A partition of the columns of a symmetrix matrix `A` is _symmetrically orthogonal_ if, for every nonzero element `A[i, j]`, either

1. the group containing the column `A[:, j]` has no other column with a nonzero in row `i`
2. the group containing the column `A[:, i]` has no other column with a nonzero in row `j`
"""
function check_symmetrically_orthogonal(
    A::AbstractMatrix, colors::AbstractVector{<:Integer}; verbose=false
)
    for i in axes(A, 2), j in axes(A, 2)
        if !iszero(A[i, j])
            group_i = filter(i2 -> (i2 != i) && (colors[i2] == colors[i]), axes(A, 2))
            group_j = filter(j2 -> (j2 != j) && (colors[j2] == colors[j]), axes(A, 2))
            A_group_i_column_j = @view A[group_i, j]
            A_group_j_column_i = @view A[group_j, i]
            nonzeros_group_i_column_j = count(!iszero, A_group_i_column_j)
            nonzeros_group_j_column_i = count(!iszero, A_group_j_column_i)
            if nonzeros_group_i_column_j > 0 && nonzeros_group_j_column_i > 0
                verbose && @warn """
                For coefficient $((i, j)), both of the following have confounding zeros:
                - color $(colors[j]) with group $group_j
                - color $(colors[i]) with group $group_i
                """
                return false
            end
        end
    end
    return true
end
