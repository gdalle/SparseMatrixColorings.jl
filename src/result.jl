## Abstract type

"""
    AbstractColoringResult

Abstract type for the result of a coloring algorithm.

It is the supertype of the object returned by the main function [`coloring`](@ref).

# Applicable methods

- [`column_colors`](@ref) and [`column_groups`](@ref) (for a `:column` or `:bidirectional` partition) 
- [`row_colors`](@ref) and [`row_groups`](@ref) (for a `:row` or `:bidirectional` partition)
- [`decompress`](@ref) and [`decompress!`](@ref)

!!! warning
    Unlike the methods above, the concrete subtypes of `AbstractColoringResult` are not part of the public API and may change without notice.
"""
abstract type AbstractColoringResult{structure,partition,decompression,M<:AbstractMatrix} end

"""
    get_matrix(result::AbstractColoringResult)

Return the matrix that was colored.
"""
function get_matrix end

"""
    column_colors(result::AbstractColoringResult)

Return a vector `color` of integer colors, one for each column of the colored matrix.
"""
function column_colors end

"""
    row_colors(result::AbstractColoringResult)

Return a vector `color` of integer colors, one for each row of the colored matrix.
"""
function row_colors end

"""
    column_groups(result::AbstractColoringResult)

Return a vector `group` such that for every color `c`, `group[c]` contains the indices of all columns that are colored with `c`.
"""
function column_groups end

"""
    row_groups(result::AbstractColoringResult)

Return a vector `group` such that for every color `c`, `group[c]` contains the indices of all rows that are colored with `c`.
"""
function row_groups end

"""
    group_by_color(color::Vector{Int})

Create `group::Vector{Vector{Int}}` such that `i ∈ group[c]` iff `color[i] == c`.

Assumes the colors are contiguously numbered from `1` to some `cmax`.
"""
function group_by_color(color::AbstractVector{<:Integer})
    cmin, cmax = extrema(color)
    @assert cmin == 1
    group = [Int[] for c in 1:cmax]
    for (k, c) in enumerate(color)
        push!(group[c], k)
    end
    return group
end

## Concrete subtypes

"""
$TYPEDEF

Default storage for the result of a coloring algorithm, containing minimal information.

# Fields

$TYPEDFIELDS

# See also

- [`AbstractColoringResult`](@ref)
"""
struct DefaultColoringResult{structure,partition,decompression,M} <:
       AbstractColoringResult{structure,partition,decompression,M}
    "matrix that was colored"
    matrix::M
    "one integer color for each column or row (depending on `partition`)"
    color::Vector{Int}
    "color groups for columns or rows (depending on `partition`)"
    group::Vector{Vector{Int}}
end

function DefaultColoringResult{structure,partition,decompression}(
    matrix::M, color::Vector{Int}
) where {structure,partition,decompression,M}
    return DefaultColoringResult{structure,partition,decompression,M}(
        matrix, color, group_by_color(color)
    )
end

get_matrix(result::DefaultColoringResult) = result.matrix

column_colors(result::DefaultColoringResult{s,:column}) where {s} = result.color
column_groups(result::DefaultColoringResult{s,:column}) where {s} = result.group

row_colors(result::DefaultColoringResult{s,:row}) where {s} = result.color
row_groups(result::DefaultColoringResult{s,:row}) where {s} = result.group
