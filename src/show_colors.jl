# Stub for Colors.jl extension in ext/SparseMatrixColoringsColorsExt.jl

"""
    show_colors(result; kwargs...)

Create a visualization for an [`AbstractColoringResult`](@ref), with the help of the [JuliaImages](https://juliaimages.org) ecosystem.

- For `:column` or `:row` colorings, it returns a couple `(A_img, B_img)`.
- For `:bidirectional` colorings, it returns a 4-tuple `(Ar_img, Ac_img, Br_img, Bc_img)`.

!!! warning
    This function is implemented in a package extension, using it requires loading [Colors.jl](https://github.com/JuliaGraphics/Colors.jl).

# Keyword arguments

- `colorscheme`: colors used for non-zero matrix entries. This can be a vector of `Colorant`s or a subsampled scheme from [ColorSchemes.jl](https://github.com/JuliaGraphics/ColorSchemes.jl).
- `background_color::Colorant`: color used for zero matrix entries and pad. Defaults to `RGBA(0,0,0,0)`, a transparent background.
- `border_color::Colorant`: color used around matrix entries. Defaults to `RGB(0, 0, 0)`, a black border.
- `scale::Int`: scale the size of matrix entries to `scale × scale` pixels. Defaults to `1`.
- `border::Int`: set border width around matrix entries, in pixles. Defaults to `0`. 
- `pad::Int`: set padding between matrix entries, in pixels. Defaults to `0`. 

For a matrix of size `(m, n)`, the resulting output will be of size `(m * (scale + 2border + pad) + pad, n * (scale + 2border + pad) + pad)`.
"""
function show_colors end
