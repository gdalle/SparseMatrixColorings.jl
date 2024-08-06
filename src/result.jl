"""
    AbstractColoringResult

Abstract type for the result of column coloring, amenable to the following methods:

- [`get_colors`](@ref)
- [`decompress_columns!`](@ref) / [`decompress_rows!`](@ref) / [`decompress_symmetric!`](@ref)
- [`decompress_columns`](@ref) / [`decompress_rows`](@ref) / [`decompress_symmetric`](@ref)
"""
abstract type AbstractColoringResult end

get_colors(result::AbstractColoringResult) = result.color

struct SimpleColoringResult <: AbstractColoringResult
    color::Vector{Int}
end

struct StarSetColoringResult <: AbstractColoringResult
    color::Vector{Int}
    star_set::StarSet
end

struct SparseColoringResult <: AbstractColoringResult
    color::Vector{Int}
    compressed_indices::Vector{Int}
end
