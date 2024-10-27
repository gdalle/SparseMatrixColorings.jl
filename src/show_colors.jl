# Stub for Colors.jl extension in ext/SparseMatrixColoringsColorsExt.jl

"""
    show_colors(result; kwargs...)

Return an image visualizing an [`AbstractColoringResult`](@ref), with the help of the the [JuliaImages](https://juliaimages.org) ecosystem.

!!! warning
    This function is implemented in a package extension, using it requires loading [Colors.jl](https://github.com/JuliaGraphics/Colors.jl).

# Keyword arguments

- `colorscheme`: colors used for non-zero matrix entries. This can be a vector of `Colorant`s or a subsampled scheme from [ColorSchemes.jl](https://github.com/JuliaGraphics/ColorSchemes.jl).
- `background::Colorant`: color used for zero matrix entries and pad. Defaults to `RGBA(0,0,0,0)`, a transparent background.
- `scale::Int`: scale the size of matrix entries to `scale × scale` pixels. Defaults to `1`. 
- `pad::Int`: set padding between matrix entries, in pixels. Defaults to `0`. 

For a matrix of size `(n, m)`, the resulting output will be of size `(n * (scale + pad) + pad, m * (scale + pad) + pad)`.
"""
function show_colors end
