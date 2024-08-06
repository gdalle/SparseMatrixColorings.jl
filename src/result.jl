"""
    AbstractColoringResult

Abstract type for the result of a (detailed) column coloring algorithm.
    
# Implemented methods

- [`get_colors`](@ref)
- [`get_groups`](@ref)
- decompression utilities
"""
abstract type AbstractColoringResult end

"""
    get_colors(result::AbstractColoringResult)

Return a vector `color` of integer colors, one per vertex.
"""
get_colors(result::AbstractColoringResult) = result.color

"""
    get_groups(result::AbstractColoringResult)

Return a vector `group` such that `group[c]` is the vector of vertices sharing color `c`.
"""
get_groups(result::AbstractColoringResult) = result.group

"""
    group_by_color(color)

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

struct SimpleColoringResult <: AbstractColoringResult
    color::Vector{Int}
    group::Vector{Vector{Int}}
end

function SimpleColoringResult(color::Vector{Int})
    return SimpleColoringResult(color, group_by_color(color))
end

struct SymmetricColoringResult <: AbstractColoringResult
    color::Vector{Int}
    group::Vector{Vector{Int}}
    star_set::StarSet
end

function SymmetricColoringResult(color::Vector{Int}, star_set::StarSet)
    return SymmetricColoringResult(color, group_by_color(color), star_set)
end

struct SparseColoringResult <: AbstractColoringResult
    color::Vector{Int}
    group::Vector{Vector{Int}}
    compressed_indices::Vector{Int}
end

function SparseColoringResult(color::Vector{Int}, compressed_indices::Vector{Int})
    return SparseColoringResult(color, group_by_color(color), compressed_indices)
end
