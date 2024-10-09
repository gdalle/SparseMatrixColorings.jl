# Visualization

SparseMatrixColorings provides some internal utilities for visualization of matrix colorings via the un-exported function [`SparseMatrixColorings.show_colors`](@ref).

!!! warning
    This function makes use of the [JuliaImages](https://juliaimages.org) ecosystem.
    Using it requires loading at least [Colors.jl](https://github.com/JuliaGraphics/Colors.jl).
    We recommend loading the full [Images.jl](https://github.com/JuliaImages/Images.jl) package for convenience, which includes Colors.jl.

## Basic usage

To obtain a visualization, simply call `show_colors` on a coloring result:

```@example img
using Images
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

!!! tip "Terminal support"
    Loading [ImageInTerminal.jl](https://github.com/JuliaImages/ImageInTerminal.jl) will allow you to show the output of `show_colors` within your terminal.
    If you use VSCode's Julia REPL, the matrix will be displayed in the plot tab.

## Customization

The visualization can be customized via keyword arguments.
The size of the matrix entries is defined by `scale`, while gaps between them are dictated by `pad`.
We recommend using the [ColorSchemes.jl](https://github.com/JuliaGraphics/ColorSchemes.jl) catalogue to customize the `colorscheme`.
Finally, a background color can be passed via the `background` keyword argument. To obtain transparent backgrounds, use the `RGBA` type.

```@example img
using ColorSchemes

julia_colors = ColorSchemes.julia
white = RGB(1, 1, 1)
show_colors(result; colorscheme=julia_colors, background=white, scale=5, pad=1)
```

## Working with large matrices

Let's demonstrate visualization of a larger random matrix:

```@example img
S = sprand(50, 50, 0.1) # sample sparse matrix

problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)
algo = GreedyColoringAlgorithm(; decompression=:direct)
result = coloring(S, problem, algo)
show_colors(result; scale=5, pad=1)
```

Instead of the default `distinguishable_colors` from Colors.jl, one can subsample a continuous colorscheme from ColorSchemes.jl:

```@example img
ncolors = maximum(column_colors(result)) # for partition=:column
colorscheme = get(ColorSchemes.rainbow, range(0.0, 1.0, length=ncolors))
show_colors(result; colorscheme=colorscheme, scale=5, pad=1)
```

## Saving images

The resulting image can be saved to a variety of formats, like PNG.
The `scale` and `pad` parameters determine the number of pixels, and thus the size of the file.

```julia
img = show_colors(result, scale=5)
save("coloring.png", img)
```

Refer to the JuliaImages [documentation on saving](https://juliaimages.org/stable/function_reference/#ref_io) for more information.
