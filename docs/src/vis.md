# Visualization

SparseMatrixColorings provides some internal utilities for visualization of matrix colorings via the un-exported function `show_colors`.

!!! warning
    This function makes use of the [Julia Images ecosystem](https://juliaimages.org/latest/).
    Using it requires loading [ColorTypes.jl](https://github.com/JuliaGraphics/ColorTypes.jl).

## Visualizing results

Currently, only `ColumnColoringResult` and `RowColoringResult` are supported.

```@example img
using ColorTypes
using SparseMatrixColorings, SparseArrays
using SparseMatrixColorings: show_colors

S = sparse([
    0 0 1 1 0 1
    1 0 0 0 1 0
    0 1 0 0 1 0
    0 1 1 0 0 0
]);

problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)
algo = GreedyColoringAlgorithm(; decompression=:direct)
result = coloring(S, problem, algo)
show_colors(result)
```

The visualization can be customized via keyword arguments:
```@example img
using ColorSchemes 
julia_colors = ColorSchemes.julia.colors
white = RGB(1, 1, 1)

show_colors(result; colorscheme=julia_colors, background=white, scale=5, padding=1)
```

!!! tip "Terminal support"
    Loading [ImageInTerminal.jl](https://github.com/JuliaImages/ImageInTerminal.jl) will allow you to show the output of `show_colors` within your terminal.

## Saving images

The [Julia Images ecosystem](https://juliaimages.org/latest/) requires you to load a separate package to save images.
[ImageIO.jl](https://github.com/JuliaIO/ImageIO.jl) is one of several options for PNG files:

```julia
using ImageIO

img = show_colors(result)
save("coloring.png", img)
```
Refer to the [Julia Images documentation](https://juliaimages.org/stable/function_reference/#ref_io) for more information.
