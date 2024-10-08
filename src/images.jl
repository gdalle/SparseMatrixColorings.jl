# Stub for Images.jl extension in ext/SparseMatrixImagesExt.jl

"""
    show_colors(result)

Return an image visualizing an [`AbstractColoringResult`](@ref).

!!! warning
    Using this function requires loading [Images.jl](https://github.com/JuliaImages/Images.jl).

## Keyword arguments
* `colorscheme`: colors used for non-zero matrix entries. Defaults to the Wong colorscheme.
* `background`: color used for zero matrix entries and padding. Defaults to a transparent background.
* `scale::Int`: scale the size of matrix entries to `scale × scale` pixels. Defaults to `1`. 
* `padding::Int`: padding between matrix entries. Defaults to `0`. 
"""
function show_colors end
