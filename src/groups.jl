"""
    color_groups(colors)

Return `groups::Vector{Vector{Int}}` such that `i ∈ groups[c]` iff `colors[i] == c`.

Assumes the colors are contiguously numbered from `1` to some `cmax`.
"""
function color_groups(colors::AbstractVector{<:Integer})
    cmin, cmax = extrema(colors)
    @assert cmin == 1
    groups = [Int[] for c in 1:cmax]
    for (k, c) in enumerate(colors)
        push!(groups[c], k)
    end
    return groups
end
