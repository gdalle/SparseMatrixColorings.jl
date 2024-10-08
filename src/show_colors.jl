# Stub for Images.jl extension in ext/SparseMatrixImagesExt.jl

"""
    show_colors(result)

Return an image visualizing an [`AbstractColoringResult`](@ref).

!!! warning
    This function makes use of the [Julia Images ecosystem](https://juliaimages.org/latest/).
    Using it requires loading [ColorTypes.jl](https://github.com/JuliaGraphics/ColorTypes.jl).


## Keyword arguments
* `colorscheme`: colors used for non-zero matrix entries. Defaults to the Wong colorscheme.
    This can be a vector of Colorants or a predefined scheme from [ColorSchemes.jl](https://github.com/JuliaGraphics/ColorSchemes.jl/blob/master/src/ColorSchemes.jl).
* `background::Colorant`: color used for zero matrix entries and pad. Defaults to `RGBA(0,0,0,0)`, a transparent background.
* `scale::Int`: scale the size of matrix entries to `scale × scale` pixels. Defaults to `1`. 
* `pad::Int`: padding between matrix entries. Defaults to `0`. 

For a matrix of size `(n, m)`, the resulting output will be of size `(n * (scale + pad) - pad, m * (scale + pad) - pad)`.
"""
function show_colors end
