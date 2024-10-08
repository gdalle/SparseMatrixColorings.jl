#=
Visualize colored matrices using the Julia Images ecosystem.
ColorTypes.jl is the most light-weight dependency to achieve this. 

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
module SparseMatrixColoringsColorTypesExt

using SparseMatrixColorings: SparseMatrixColorings
using SparseMatrixColorings: AbstractColoringResult, sparsity_pattern, column_colors, row_colors
using ColorTypes: Colorant, RGB, RGBA

# Default to Makie.jl's default color scheme "Wong":
# https://github.com/MakieOrg/Makie.jl/blob/e90c042d16b461e67b750e5ce53790e732281dba/src/theming.jl#L1-L16
# Conservative 7-color palette from Points of view: Color blindness, Bang Wong - Nature Methods
# https://www.nature.com/articles/nmeth.1618?WT.ec_id=NMETH-201106
const DEFAULT_COLOR_SCHEME = [
    RGB(0 / 255, 114 / 255, 178 / 255),   # blue
    RGB(230 / 255, 159 / 255, 0 / 255),   # orange
    RGB(0 / 255, 158 / 255, 115 / 255),   # green
    RGB(204 / 255, 121 / 255, 167 / 255), # reddish purple
    RGB(86 / 255, 180 / 255, 233 / 255),  # sky blue
    RGB(213 / 255, 94 / 255, 0 / 255),    # vermillion
    RGB(240 / 255, 228 / 255, 66 / 255),  # yellow
]
const DEFAULT_BACKGROUND = RGBA(0, 0, 0, 0)
const DEFAULT_SCALE = 1   # update docstring in src/images.jl when changing this default
const DEFAULT_PAD = 0 # update docstring in src/images.jl when changing this default

ncolors(res::AbstractColoringResult{s,:column}) where {s} = maximum(column_colors(res))
ncolors(res::AbstractColoringResult{s,:row}) where {s} = maximum(row_colors(res))

## Top-level function that handles argument errors, eagerly promotes types and allocates output buffer

function SparseMatrixColorings.show_colors(
    res::AbstractColoringResult;
    colorscheme=DEFAULT_COLOR_SCHEME,
    background::Colorant=DEFAULT_BACKGROUND, # color used for zero matrix entries and pad
    scale::Int=DEFAULT_SCALE, # scale size of matrix entries to `scale × scale` pixels
    pad::Int=DEFAULT_PAD, # pad between matrix entries
    warn::Bool=true,
)
    if warn && ncolors(res) > length(colorscheme)
        # TODO: add option to cycle colors if colorscheme is too short?
        @warn "`show_colors` will reuse colors since the provided color scheme is smaller than the $(ncolors(res)) matrix colors.
        You can turn this warning off via the keyword argument `warn = false`"
    end
    scale < 1 && error("keyword-argument `scale` has to be ≥ 1")
    pad < 0 && error("keyword-argument `pad` has to be ≥ 0")

    colorscheme, background = promote_colors(colorscheme, background)
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
    Base.require_one_based_indexing(res.A)
    hi, wi = size(res.A)
    h = hi * (scale + pad) - pad
    w = wi * (scale + pad) - pad
    return fill(background, h, w)
end

## Implementations for different AbstractColoringResult types start here

function show_colors!(
    out, res::AbstractColoringResult{s,:column}, colorscheme, scale, pad
) where {s}
    stencil = CartesianIndices((1:scale, 1:scale))
    color_indices = mod1.(res.color, length(colorscheme)) # cycle color indices if necessary
    column_colors = colorscheme[color_indices]

    pattern = sparsity_pattern(res)
    for I in CartesianIndices(pattern)
        if !iszero(pattern[I])
            r, c = Tuple(I)
            area = (I - CartesianIndex(1, 1)) * (scale + pad) .+ stencil # one matrix entry
            out[area] .= column_colors[c]
        end
    end
    return out
end

function show_colors!(
    out, res::AbstractColoringResult{s,:row}, colorscheme, scale, pad
) where {s}
    stencil = CartesianIndices((1:scale, 1:scale))
    color_indices = mod1.(res.color, length(colorscheme)) # cycle color indices if necessary
    row_colors = colorscheme[color_indices]

    pattern = sparsity_pattern(res)
    for I in CartesianIndices(pattern)
        if !iszero(pattern[I])
            r, c = Tuple(I)
            area = (I - CartesianIndex(1, 1)) * (scale + pad) .+ stencil # one matrix entry
            out[area] .= row_colors[r]
        end
    end
    return out
end

end # module
