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
    ncolors,
    compress
using Colors: Colorant, RGB, RGBA, distinguishable_colors

# update docstring in src/images.jl when changing this default
const DEFAULT_BACKGROUND_COLOR = RGBA(0, 0, 0, 0)
const DEFAULT_BORDER_COLOR = RGB(0, 0, 0)
const DEFAULT_SCALE = 1
const DEFAULT_BORDER = 0
const DEFAULT_PAD = 0

## Top-level function that handles argument errors, eagerly promotes types and allocates output buffer

function SparseMatrixColorings.show_colors(
    res::AbstractColoringResult;
    colorscheme=nothing,
    background_color::Colorant=DEFAULT_BACKGROUND_COLOR, # color used for zero matrix entries and pad
    border_color::Colorant=DEFAULT_BORDER_COLOR, # color used for zero matrix entries and pad
    scale::Int=DEFAULT_SCALE, # scale size of matrix entries to `scale × scale` pixels
    border::Int=DEFAULT_BORDER,  # border around matrix entries
    pad::Int=DEFAULT_PAD, # pad between matrix entries
    warn::Bool=true,
)
    scale < 1 && throw(ArgumentError("`scale` has to be ≥ 1."))
    border < 0 && throw(ArgumentError("`border` has to be ≥ 0."))
    pad < 0 && throw(ArgumentError("`pad` has to be ≥ 0."))

    if !isnothing(colorscheme)
        if warn && ncolors(res) > length(colorscheme)
            @warn "`show_colors` will reuse colors since the provided `colorscheme` has $(length(colorscheme)) colors and the matrix needs $(ncolors(res)). You can turn off this warning via the keyword argument `warn = false`, or choose a larger `colorscheme` from ColorSchemes.jl."
        end
        colorscheme, background_color, border_color = promote_colors(
            colorscheme, background_color, border_color
        )
    else
        # Sample n distinguishable colors, excluding the background and border color
        colorscheme = distinguishable_colors(
            ncolors(res),
            [convert(RGB, background_color), convert(RGB, border_color)];
            dropseed=true,
        )
    end
    outs = allocate_outputs(res, background_color, border_color, scale, border, pad)
    return show_colors!(outs..., res, colorscheme, scale, border, pad)
end

function promote_colors(colorscheme, background_color, border_color)
    # eagerly promote colors to same type
    T = promote_type(eltype(colorscheme), typeof(background_color), typeof(border_color))
    colorscheme = convert.(T, colorscheme)
    background_color = convert(T, background_color)
    border_color = convert(T, border_color)
    return colorscheme, background_color, border_color
end

# Given a CartesianIndex I of an entry in the original matrix, 
# this function returns the corresponding area in the output image as CartesianIndices.
function matrix_entry_area(I::CartesianIndex, scale, border, pad)
    stencil = CartesianIndices((1:scale, 1:scale))
    return CartesianIndex(1, 1) * (border + pad) +
           (I - CartesianIndex(1, 1)) * (scale + 2border + pad) .+ stencil
end

function matrix_entry_plus_border_area(I::CartesianIndex, scale, border, pad)
    stencil = CartesianIndices((1:(scale + 2border), 1:(scale + 2border)))
    return CartesianIndex(1, 1) * pad +
           (I - CartesianIndex(1, 1)) * (scale + 2border + pad) .+ stencil
end

function allocate_outputs(
    res::Union{AbstractColoringResult{s,:column},AbstractColoringResult{s,:row}},
    background_color::Colorant,
    border_color::Colorant,
    scale::Int,
    border::Int,
    pad::Int,
) where {s}
    A = sparsity_pattern(res)
    B = compress(A, res)
    Base.require_one_based_indexing(A)
    Base.require_one_based_indexing(B)
    hA, wA = size(A) .* (scale + 2border + pad) .+ (pad)
    hB, wB = size(B) .* (scale + 2border + pad) .+ (pad)
    A_img = fill(background_color, hA, wA)
    B_img = fill(background_color, hB, wB)
    for I in CartesianIndices(A)
        if !iszero(A[I])
            area = matrix_entry_area(I, scale, border, pad)
            barea = matrix_entry_plus_border_area(I, scale, border, pad)
            A_img[barea] .= border_color
            A_img[area] .= background_color
        end
    end
    for I in CartesianIndices(B)
        if !iszero(B[I])
            area = matrix_entry_area(I, scale, border, pad)
            barea = matrix_entry_plus_border_area(I, scale, border, pad)
            B_img[barea] .= border_color
            B_img[area] .= background_color
        end
    end
    return A_img, B_img
end

function allocate_outputs(
    res::AbstractColoringResult{s,:bidirectional},
    background_color::Colorant,
    border_color::Colorant,
    scale::Int,
    border::Int,
    pad::Int,
) where {s}
    A = sparsity_pattern(res)
    Br, Bc = compress(A, res)
    Base.require_one_based_indexing(A)
    Base.require_one_based_indexing(Br)
    Base.require_one_based_indexing(Bc)
    hA, wA = size(A) .* (scale + 2border + pad) .+ (pad)
    hBr, wBr = size(Br) .* (scale + 2border + pad) .+ (pad)
    hBc, wBc = size(Bc) .* (scale + 2border + pad) .+ (pad)
    Arc_img = fill(background_color, hA, wA)
    Ar_img = fill(background_color, hA, wA)
    Ac_img = fill(background_color, hA, wA)
    Br_img = fill(background_color, hBr, wBr)
    Bc_img = fill(background_color, hBc, wBc)
    for I in CartesianIndices(A)
        if !iszero(A[I])
            area = matrix_entry_area(I, scale, border, pad)
            barea = matrix_entry_plus_border_area(I, scale, border, pad)
            Arc_img[barea] .= border_color
            Ar_img[barea] .= border_color
            Ac_img[barea] .= border_color
            Arc_img[area] .= background_color
            Ar_img[area] .= background_color
            Ac_img[area] .= background_color
        end
    end
    for I in CartesianIndices(Br)
        if !iszero(Br[I])
            area = matrix_entry_area(I, scale, border, pad)
            barea = matrix_entry_plus_border_area(I, scale, border, pad)
            Br_img[barea] .= border_color
            Br_img[area] .= background_color
        end
    end
    for I in CartesianIndices(Bc)
        if !iszero(Bc[I])
            area = matrix_entry_area(I, scale, border, pad)
            barea = matrix_entry_plus_border_area(I, scale, border, pad)
            Bc_img[barea] .= border_color
            Bc_img[area] .= background_color
        end
    end
    return Arc_img, Ar_img, Ac_img, Br_img, Bc_img
end

## Implementations for different AbstractColoringResult types start here

function show_colors!(
    A_img::AbstractMatrix{<:Colorant},
    B_img::AbstractMatrix{<:Colorant},
    res::AbstractColoringResult{s,:column},
    colorscheme,
    scale::Int,
    border::Int,
    pad::Int,
) where {s}
    # cycle color indices if necessary
    A_color_indices = mod1.(column_colors(res), length(colorscheme))
    B_color_indices = mod1.(1:ncolors(res), length(colorscheme))
    A_colors = colorscheme[A_color_indices]
    B_colors = colorscheme[B_color_indices]
    A = sparsity_pattern(res)
    B = compress(A, res)
    for I in CartesianIndices(A)
        if !iszero(A[I])
            r, c = Tuple(I)
            area = matrix_entry_area(I, scale, border, pad)
            if column_colors(res)[c] > 0
                A_img[area] .= A_colors[c]
            end
        end
    end
    for I in CartesianIndices(B)
        if !iszero(B[I])
            r, c = Tuple(I)
            area = matrix_entry_area(I, scale, border, pad)
            B_img[area] .= B_colors[c]
        end
    end
    return A_img, B_img
end

function show_colors!(
    A_img::AbstractMatrix{<:Colorant},
    B_img::AbstractMatrix{<:Colorant},
    res::AbstractColoringResult{s,:row},
    colorscheme,
    scale::Int,
    border::Int,
    pad::Int,
) where {s}
    # cycle color indices if necessary
    A_color_indices = mod1.(row_colors(res), length(colorscheme))
    B_color_indices = mod1.(1:ncolors(res), length(colorscheme))
    A_colors = colorscheme[A_color_indices]
    B_colors = colorscheme[B_color_indices]
    A = sparsity_pattern(res)
    B = compress(A, res)
    for I in CartesianIndices(A)
        if !iszero(A[I])
            r, c = Tuple(I)
            area = matrix_entry_area(I, scale, border, pad)
            if row_colors(res)[r] > 0
                A_img[area] .= A_colors[r]
            end
        end
    end
    for I in CartesianIndices(B)
        if !iszero(B[I])
            r, c = Tuple(I)
            area = matrix_entry_area(I, scale, border, pad)
            B_img[area] .= B_colors[r]
        end
    end
    return A_img, B_img
end

mytriu(area) = [area[i, j] for i in axes(area, 1) for j in axes(area, 2) if i < j]
mytril(area) = [area[i, j] for i in axes(area, 1) for j in axes(area, 2) if i > j]

function show_colors!(
    Arc_img::AbstractMatrix{<:Colorant},
    Ar_img::AbstractMatrix{<:Colorant},
    Ac_img::AbstractMatrix{<:Colorant},
    Br_img::AbstractMatrix{<:Colorant},
    Bc_img::AbstractMatrix{<:Colorant},
    res::AbstractColoringResult{s,:bidirectional},
    colorscheme,
    scale::Int,
    border::Int,
    pad::Int,
) where {s}
    scale < 3 && throw(ArgumentError("`scale` has to be ≥ 3 to visualize bicoloring"))
    # cycle color indices if necessary
    row_shift = maximum(column_colors(res))
    A_ccolor_indices = mod1.(column_colors(res), length(colorscheme))
    A_rcolor_indices = mod1.(row_shift .+ row_colors(res), length(colorscheme))
    B_ccolor_indices = mod1.(1:maximum(column_colors(res)), length(colorscheme))
    B_rcolor_indices =
        mod1.((row_shift + 1):(row_shift + maximum(row_colors(res))), length(colorscheme))
    A_ccolors = colorscheme[A_ccolor_indices]
    A_rcolors = colorscheme[A_rcolor_indices]
    B_ccolors = colorscheme[B_ccolor_indices]
    B_rcolors = colorscheme[B_rcolor_indices]
    A = sparsity_pattern(res)
    Br, Bc = compress(A, res)
    for I in CartesianIndices(A)
        if !iszero(A[I])
            r, c = Tuple(I)
            area = matrix_entry_area(I, scale, border, pad)
            if column_colors(res)[c] > 0
                Arc_img[mytriu(area)] .= A_ccolors[c]
                Ac_img[area] .= A_ccolors[c]
            end
            if row_colors(res)[r] > 0
                Arc_img[mytril(area)] .= A_rcolors[r]
                Ar_img[area] .= A_rcolors[r]
            end
        end
    end
    for I in CartesianIndices(Br)
        if !iszero(Br[I])
            r, c = Tuple(I)
            area = matrix_entry_area(I, scale, border, pad)
            Br_img[area] .= B_rcolors[r]
        end
    end
    for I in CartesianIndices(Bc)
        if !iszero(Bc[I])
            r, c = Tuple(I)
            area = matrix_entry_area(I, scale, border, pad)
            Bc_img[area] .= B_ccolors[c]
        end
    end
    return Arc_img, Ar_img, Ac_img, Br_img, Bc_img
end

end # module
