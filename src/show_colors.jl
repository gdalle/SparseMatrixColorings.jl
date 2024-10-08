# Stub for Images.jl extension in ext/SparseMatrixImagesExt.jl

"""
    show_colors(result)

Return an image visualizing an [`AbstractColoringResult`](@ref).

!!! warning
    This function makes use of the [Julia Images ecosystem](https://juliaimages.org/latest/).
    Using it requires loading [ColorTypes.jl](https://github.com/JuliaGraphics/ColorTypes.jl).


## Keyword arguments
* `colorscheme`: colors used for non-zero matrix entries. Defaults to the Wong colorscheme.
* `background`: color used for zero matrix entries and pad. Defaults to a transparent background.
* `scale::Int`: scale the size of matrix entries to `scale × scale` pixels. Defaults to `1`. 
* `pad::Int`: padding between matrix entries. Defaults to `0`. 

For a matrix of size `(n, m)`, the resulting output will be of size `(n * (scale + pad) - pad, m * (scale + pad) - pad)`.
"""
function show_colors end
