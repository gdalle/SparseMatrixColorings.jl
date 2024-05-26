"""
    color_groups(color)

Return `group::Vector{Vector{Int}}` such that `i ∈ group[c]` iff `color[i] == c`.

Assumes the colors are contiguously numbered from `1` to some `cmax`.
"""
function color_groups(color::AbstractVector{<:Integer})
    cmin, cmax = extrema(color)
    @assert cmin == 1
    group = [Int[] for c in 1:cmax]
    for (k, c) in enumerate(color)
        push!(group[c], k)
    end
    return group
end
