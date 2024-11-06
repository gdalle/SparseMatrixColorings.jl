#=
Visualize colored matrices using the Julia Images ecosystem.
Colors.jl is nearly the most light-weight dependency to achieve this. 

This code is written prioritizing maintainability over performance

How it works
≡≡≡≡≡≡≡≡≡≡≡≡
- First, the outer `show_colors` function gets called, which
    - handles argument errors
    - eagerly promotes color types in `promote_colors` to support transparency
    - allocates an output buffer in `allocate_output`
- The allocated output is a matrix filled with the background color
- An internal `show_color!` function is called on the allocated output
    - for each non-zero entry in the coloring, the output is filled in
=#
module SparseMatrixColoringsColorsExt

using SparseMatrixColorings:
    SparseMatrixColorings,
    AbstractColoringResult,
    sparsity_pattern,
    column_colors,
    row_colors,
    ncolors
using Colors: Colorant, RGB, RGBA, distinguishable_colors

const DEFAULT_BACKGROUND = RGBA(0, 0, 0, 0)
const DEFAULT_SCALE = 1   # update docstring in src/images.jl when changing this default
const DEFAULT_PAD = 0 # update docstring in src/images.jl when changing this default

# Sample n distinguishable colors, excluding the background color
default_colorscheme(n, background) = distinguishable_colors(n, background; dropseed=true)

## Top-level function that handles argument errors, eagerly promotes types and allocates output buffer

function SparseMatrixColorings.show_colors(
    res::AbstractColoringResult;
    colorscheme=nothing,
    background::Colorant=DEFAULT_BACKGROUND, # color used for zero matrix entries and pad
    scale::Int=DEFAULT_SCALE, # scale size of matrix entries to `scale × scale` pixels
    pad::Int=DEFAULT_PAD, # pad between matrix entries
    warn::Bool=true,
)
    scale < 1 && throw(ArgumentError("`scale` has to be ≥ 1."))
    pad < 0 && throw(ArgumentError("`pad` has to be ≥ 0."))

    if !isnothing(colorscheme)
        if warn && ncolors(res) > length(colorscheme)
            @warn "`show_colors` will reuse colors since the provided `colorscheme` has $(length(colorscheme)) colors and the matrix needs $(ncolors(res)). You can turn off this warning via the keyword argument `warn = false`, or choose a larger `colorscheme` from ColorSchemes.jl."
        end
        colorscheme, background = promote_colors(colorscheme, background)
    else
        colorscheme = default_colorscheme(ncolors(res), convert(RGB, background))
    end
    out = allocate_output(res, background, scale, pad)
    return show_colors!(out, res, colorscheme, scale, pad)
end

function promote_colors(colorscheme, background)
    # eagerly promote colors to same type
    T = promote_type(eltype(colorscheme), typeof(background))
    colorscheme = convert.(T, colorscheme)
    background = convert(T, background)
    return colorscheme, background
end

function allocate_output(
    res::AbstractColoringResult, background::Colorant, scale::Int, pad::Int
)
    A = sparsity_pattern(res)
    Base.require_one_based_indexing(A)
    hi, wi = size(A)
    h = hi * (scale + pad) + pad
    w = wi * (scale + pad) + pad
    return fill(background, h, w)
end

# Given a CartesianIndex I of an entry in the original matrix, 
# this function returns the corresponding area in the output image as CartesianIndices.
function matrix_entry_area(I::CartesianIndex, scale, pad)
    stencil = CartesianIndices((1:scale, 1:scale))
    return CartesianIndex(pad, pad) + (I - CartesianIndex(1, 1)) * (scale + pad) .+ stencil
end

## Implementations for different AbstractColoringResult types start here

function show_colors!(
    out, res::AbstractColoringResult{s,:column}, colorscheme, scale, pad
) where {s}
    color_indices = mod1.(column_colors(res), length(colorscheme)) # cycle color indices if necessary
    colors = colorscheme[color_indices]
    pattern = sparsity_pattern(res)
    for I in CartesianIndices(pattern)
        if !iszero(pattern[I])
            r, c = Tuple(I)
            area = matrix_entry_area(I, scale, pad)
            out[area] .= colors[c]
        end
    end
    return out
end

function show_colors!(
    out, res::AbstractColoringResult{s,:row}, colorscheme, scale, pad
) where {s}
    color_indices = mod1.(row_colors(res), length(colorscheme)) # cycle color indices if necessary
    colors = colorscheme[color_indices]
    pattern = sparsity_pattern(res)
    for I in CartesianIndices(pattern)
        if !iszero(pattern[I])
            r, c = Tuple(I)
            area = matrix_entry_area(I, scale, pad)
            out[area] .= colors[r]
        end
    end
    return out
end

function show_colors!(
    out, res::AbstractColoringResult{s,:bidirectional}, colorscheme, scale, pad
) where {s}
    ccolor_indices = mod1.(column_colors(res), length(colorscheme)) # cycle color indices if necessary
    row_shift = maximum(column_colors(res))
    rcolor_indices = mod1.(row_shift .+ row_colors(res), length(colorscheme)) # cycle color indices if necessary
    ccolors = colorscheme[ccolor_indices]
    rcolors = colorscheme[rcolor_indices]
    pattern = sparsity_pattern(res)
    for I in CartesianIndices(pattern)
        if !iszero(pattern[I])
            r, c = Tuple(I)
            area = matrix_entry_area(I, scale, pad)
            for i in axes(area, 1), j in axes(area, 2)
                if j > i
                    out[area[i, j]] = ccolors[c]
                elseif i > j
                    out[area[i, j]] = rcolors[r]
                end
            end
        end
    end
    return out
end

end # module
