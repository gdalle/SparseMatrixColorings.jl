# Visualization

SparseMatrixColorings provides some internal utilities for visualization of matrix colorings via the un-exported function `show_colors`.

!!! warning
    This function makes use of the [Julia Images ecosystem](https://juliaimages.org/latest/).
    Using it requires loading [ColorTypes.jl](https://github.com/JuliaGraphics/ColorTypes.jl).

## Basic usage

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

## Customization

The visualization can be customized via keyword arguments.
By setting `pad` to a value higher than `0`, gaps are added between matrix entries.
The size of the entries can be customized via `scale`. This parameter is also useful to upscale PNGs for later saving (see below).

We recommend using the [ColorSchemes.jl catalogue](https://juliagraphics.github.io/ColorSchemes.jl/dev/catalogue/) to customize the `colorscheme`.
Finally, a background color can be passed via the `background` keyword argument. To obtain transparent backgrounds, use the `RGBA` type.

```@example img
using ColorSchemes 
julia_colors = ColorSchemes.julia
white = RGB(1, 1, 1)

show_colors(result; colorscheme=julia_colors, background=white, scale=5, pad=1)
```

!!! tip "Terminal support"
    Loading [ImageInTerminal.jl](https://github.com/JuliaImages/ImageInTerminal.jl) will allow you to show the output of `show_colors` within your terminal.

## Working with large matrices

When working with large matrices, the default color scheme might be smaller than the number of matrix colors.
In this case, a warning will be thrown, notifying you of a reuse of colors.
To disable this warning, call `show_colors` with `warn=false`.

Let's demonstrate this by sampling a larger random matrix:
```@example img
S = sprand(50, 50, 0.1) # sample sparse matrix

problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)
algo = GreedyColoringAlgorithm(; decompression=:direct)
result = coloring(S, problem, algo)
show_colors(result; warn=false, scale=5, pad=1)
```

Since this visualization is ambiguous, we instead recomment subsampling a continuous colorscheme from ColorSchemes.jl:
```@example img
ncolors = maximum(column_colors(result)) # for partition=:column
colorscheme = get(ColorSchemes.rainbow, range(0.0, 1.0, length=ncolors))
show_colors(result; colorscheme=colorscheme, scale=5, pad=1)
```

Using [Colors.jl](https://github.com/JuliaGraphics/Colors.jl) and ColorSchemes, you can also generate `distinguishable_colors`:
```@example img
using Colors, ColorSchemes
colorscheme = distinguishable_colors(ncolors, transform=protanopic)
show_colors(result; colorscheme=colorscheme, scale=5, pad=1)
```

## Saving images

The [Julia Images ecosystem](https://juliaimages.org/latest/) requires you to load a separate package to save images.
[ImageIO.jl](https://github.com/JuliaIO/ImageIO.jl) is one of several options for PNG files:

```julia
using ImageIO

img = show_colors(result)
save("coloring.png", img)
```

Refer to the [Julia Images documentation](https://juliaimages.org/stable/function_reference/#ref_io) for more information.