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
    outs = allocate_outputs(res, background, scale, pad)
    return show_colors!(outs..., res, colorscheme, scale, pad)
end

function promote_colors(colorscheme, background)
    # eagerly promote colors to same type
    T = promote_type(eltype(colorscheme), typeof(background))
    colorscheme = convert.(T, colorscheme)
    background = convert(T, background)
    return colorscheme, background
end

function allocate_outputs(
    res::Union{AbstractColoringResult{s,:column},AbstractColoringResult{s,:row}},
    background::Colorant,
    scale::Int,
    pad::Int,
) where {s}
    A = sparsity_pattern(res)
    B = compress(A, res)
    Base.require_one_based_indexing(A)
    Base.require_one_based_indexing(B)
    hA, wA = size(A) .* (scale + pad) .+ pad
    hB, wB = size(B) .* (scale + pad) .+ pad
    A_img = fill(background, hA, wA)
    B_img = fill(background, hB, wB)
    return A_img, B_img
end

function allocate_outputs(
    res::AbstractColoringResult{s,:bidirectional},
    background::Colorant,
    scale::Int,
    pad::Int,
) where {s}
    A = sparsity_pattern(res)
    Br, Bc = compress(A, res)
    Base.require_one_based_indexing(A)
    Base.require_one_based_indexing(Br)
    Base.require_one_based_indexing(Bc)
    hA, wA = size(A) .* (scale + pad) .+ pad
    hBr, wBr = size(Br) .* (scale + pad) .+ pad
    hBc, wBc = size(Bc) .* (scale + pad) .+ pad
    A_img = fill(background, hA, wA)
    Br_img = fill(background, hBr, wBr)
    Bc_img = fill(background, hBc, wBc)
    return A_img, Br_img, Bc_img
end

# Given a CartesianIndex I of an entry in the original matrix, 
# this function returns the corresponding area in the output image as CartesianIndices.
function matrix_entry_area(I::CartesianIndex, scale, pad)
    stencil = CartesianIndices((1:scale, 1:scale))
    return CartesianIndex(pad, pad) + (I - CartesianIndex(1, 1)) * (scale + pad) .+ stencil
end

## Implementations for different AbstractColoringResult types start here

function show_colors!(
    A_img::AbstractMatrix{<:Colorant},
    B_img::AbstractMatrix{<:Colorant},
    res::AbstractColoringResult{s,:column},
    colorscheme,
    scale,
    pad,
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
            area = matrix_entry_area(I, scale, pad)
            if column_colors(res)[c] > 0
                A_img[area] .= A_colors[c]
            end
        end
    end
    for I in CartesianIndices(B)
        if !iszero(B[I])
            r, c = Tuple(I)
            area = matrix_entry_area(I, scale, pad)
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
    scale,
    pad,
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
            area = matrix_entry_area(I, scale, pad)
            if row_colors(res)[r] > 0
                A_img[area] .= A_colors[r]
            end
        end
    end
    for I in CartesianIndices(B)
        if !iszero(B[I])
            r, c = Tuple(I)
            area = matrix_entry_area(I, scale, pad)
            B_img[area] .= B_colors[r]
        end
    end
    return A_img, B_img
end

function show_colors!(
    A_img::AbstractMatrix{<:Colorant},
    Br_img::AbstractMatrix{<:Colorant},
    Bc_img::AbstractMatrix{<:Colorant},
    res::AbstractColoringResult{s,:bidirectional},
    colorscheme,
    scale,
    pad,
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
            area = matrix_entry_area(I, scale, pad)
            for i in axes(area, 1), j in axes(area, 2)
                if j > i
                    if column_colors(res)[c] > 0
                        A_img[area[i, j]] = A_ccolors[c]
                    end
                elseif i > j
                    if row_colors(res)[r] > 0
                        A_img[area[i, j]] = A_rcolors[r]
                    end
                end
            end
        end
    end
    for I in CartesianIndices(Br)
        if !iszero(Br[I])
            r, c = Tuple(I)
            area = matrix_entry_area(I, scale, pad)
            Br_img[area] .= B_rcolors[r]
        end
    end
    for I in CartesianIndices(Bc)
        if !iszero(Bc[I])
            r, c = Tuple(I)
            area = matrix_entry_area(I, scale, pad)
            Bc_img[area] .= B_ccolors[c]
        end
    end
    return A_img, Br_img, Bc_img
end

end # module
