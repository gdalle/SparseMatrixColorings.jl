# Stub for Images.jl extension in ext/SparseMatrixImagesExt.jl

"""
    show_colors(result)

Return an image visualizing an [`AbstractColoringResult`](@ref).

!!! warning
    This feature requires first loading the Images.jl package.

## Keyword arguments
* `colorscheme`: colors used for non-zero matrix entries. 
* `background`: color used for zero matrix entries and padding. Defaults to a transparent background.
* `scale::Int`: scale the size of matrix entries to `scale × scale` pixels. Defaults to `5`. 
* `padding::Int`: padding between matrix entries. Defaults to `1`. 
"""
function show_colors end
