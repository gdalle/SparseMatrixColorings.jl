# Visualize colored matrices using Images.jl
module SparseMatrixColoringsImagesExt

if isdefined(Base, :get_extension)
    using SparseMatrixColorings:
        SparseMatrixColorings,
        AbstractColoringResult,
        ColumnColoringResult,
        RowColoringResult
    using Images: Colorant, RGB, RGBA
else
    using ..SparseMatrixColorings:
        SparseMatrixColorings,
        AbstractColoringResult,
        ColumnColoringResult,
        RowColoringResult
    using ..Images: Colorant, RGB, RGBA
end

# Default to Makie.jl's default color scheme "Wong":
# https://github.com/MakieOrg/Makie.jl/blob/e90c042d16b461e67b750e5ce53790e732281dba/src/theming.jl#L1-L16
# Conservative 7-color palette from Points of view: Color blindness, Bang Wong - Nature Methods
# https://www.nature.com/articles/nmeth.1618?WT.ec_id=NMETH-201106
DEFAULT_COLOR_SCHEME = [
    RGB(0 / 255, 114 / 255, 178 / 255),   # blue
    RGB(230 / 255, 159 / 255, 0 / 255),   # orange
    RGB(0 / 255, 158 / 255, 115 / 255),   # green
    RGB(204 / 255, 121 / 255, 167 / 255), # reddish purple
    RGB(86 / 255, 180 / 255, 233 / 255),  # sky blue
    RGB(213 / 255, 94 / 255, 0 / 255),    # vermillion
    RGB(240 / 255, 228 / 255, 66 / 255),  # yellow
]
DEFAULT_BACKGROUND = RGBA(0, 0, 0, 0)
const DEFAULT_SCALE = 1   # update docstring in src/images.jl when changing this default
const DEFAULT_PADDING = 0 # update docstring in src/images.jl when changing this default

ncolors(res::AbstractColoringResult) = maximum(res.color)

## Top-level function that handles argument errors, eagerly promotes types and allocates output buffer

function SparseMatrixColorings.show_colors(
    res::AbstractColoringResult;
    colorscheme=DEFAULT_COLOR_SCHEME,
    background::Colorant=DEFAULT_BACKGROUND, # color used for zero matrix entries and padding
    scale::Int=DEFAULT_SCALE, # scale size of matrix entries to `scale × scale` pixels
    padding::Int=DEFAULT_PADDING, # padding between matrix entries
)
    ncolors(res) > length(colorscheme) &&
    # TODO: add option to cycle colors if colorscheme is too short?
        error(
            "`colorscheme` with $(length(colorscheme)) is too small to represent matrix coloring with $(ncolors(res)) colors.",
        )
    scale < 1 && error("keyword-argument `scale` has to be ≥ 1")
    padding < 0 && error("keyword-argument `padding` has to be ≥ 0")

    colorscheme, background = _promote_colors(colorscheme, background)
    out = _allocate_output(res, background, scale, padding)
    return _show_colors!(out, res, colorscheme, scale, padding)
end

function _promote_colors(colorscheme, background)
    # eagerly promote colors to same type
    T = promote_type(eltype(colorscheme), typeof(background))
    colorscheme = convert.(T, colorscheme)
    background = convert(T, background)
    return colorscheme, background
end

function _allocate_output(
    res::AbstractColoringResult, background::Colorant, scale::Int, padding::Int
)
    Base.require_one_based_indexing(res.A)
    hi, wi = size(res.A)
    h = hi * (scale + padding) - padding
    w = wi * (scale + padding) - padding
    return fill(background, h, w)
end

## Implementations for different AbstractColoringResult types start here

function _show_colors!(out, res::AbstractColoringResult, colorscheme, scale, padding)
    return error(
        "`show_colors` is currently only implemented for `ColumnColoringResult` and `RowColoringResult`.",
    )
end

function _show_colors!(out, res::ColumnColoringResult, colorscheme, scale, padding)
    stencil = CartesianIndices((1:scale, 1:scale))
    column_colors = colorscheme[res.color]
    for I in CartesianIndices(res.A)
        if !iszero(res.A[I])
            r, c = Tuple(I)
            area = (I - CartesianIndex(1, 1)) * (scale + padding) .+ stencil # one matrix entry
            out[area] .= column_colors[c]
        end
    end
    return out
end

function _show_colors!(out, res::RowColoringResult, colorscheme, scale, padding)
    stencil = CartesianIndices((1:scale, 1:scale))
    row_colors = colorscheme[res.color]
    for I in CartesianIndices(res.A)
        if !iszero(res.A[I])
            r, c = Tuple(I)
            area = (I - CartesianIndex(1, 1)) * (scale + padding) .+ stencil # one matrix entry
            out[area] .= row_colors[r]
        end
    end
    return out
end

end # module
