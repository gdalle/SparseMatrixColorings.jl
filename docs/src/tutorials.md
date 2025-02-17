# Tutorials

## Introduction to SparseMatrixColorings.jl

The three main functions to perform a coloring with `SparseMatrixColorings.jl` are [`coloring`](@ref coloring), [`ColoringProblem`](@ref ColoringProblem) and [`GreedyColoringAlgorithm`](@ref GreedyColoringAlgorithm).

```@example tutorial; continued = true
using SparseMatrixColorings, SparseArrays

S = sparse([
  0 0 1 1 0 1
  1 0 0 0 1 0
  0 1 0 0 1 0
  0 1 1 0 0 0
])

problem = ColoringProblem()
algo = GreedyColoringAlgorithm()
result = coloring(S, problem, algo)
```

Based on the result of `coloring`, you can easily recover a vector of integer colors with [`row_colors`](@ref row_colors), [`column_colors`](@ref column_colors), as well as the groups of colors with [`row_groups`](@ref row_groups) and [`column_groups`](@ref column_groups).

```@example tutorial
column_colors(result)
```

```@example tutorial
column_groups(result)
```

The number of colors can be determined with [`ncolors`](@ref ncolors).
```@example tutorial
ncolors(result)
```

The functions [`compress`](@ref compress) and [`decompress`](@ref decompress) efficiently store and retrieve compressed representations of colorings for sparse matrices.

```@example tutorial
M = sparse([
  0 0 4 6 0 9
  1 0 0 0 7 0
  0 2 0 0 8 0
  0 3 5 0 0 0
])

B = compress(M, result)
```

```@example tutorial
C = decompress(B, result)
```

The functions [`decompress!`](@ref decompress!) and [`decompress_single_color!`](@ref decompress_single_color!) are in-place variants of [`decompress`](@ref decompress).

```@example tutorial
D = [10  14  18
     11  15  0
     12  16  0
     13  17  0]

decompress!(C, D, result)
```
```@example tutorial
C .= 0
decompress_single_color!(C, D[:, 2], 2, result)
```

We now illustrate the six variants of colorings available in `SparseMatrixColorings.jl` on the following matrix with a symmetric sparsity pattern.

```@example tutorial
using SparseMatrixColorings: show_colors # hide
using Images # hide
using MatrixMarket
S = MatrixMarket.mmread("smc.mtx")
```

## Column coloring

```@example tutorial; continued = true
problem = ColoringProblem(; structure=:nonsymmetric,
                            partition=:column)

order = NaturalOrder()

algo = GreedyColoringAlgorithm(order;
                               decompression=:direct)

result = coloring(S, problem, algo)
```
```@example tutorial
ccolors = column_colors(result)
cgroups = column_groups(result)
num_colors = ncolors(result)
```

```@example tutorial
A_img, B_img = show_colors(result, scale=5, pad=2) # hide
A_img # hide
```

## Row coloring

```@example tutorial; continued = true
problem = ColoringProblem(; structure=:nonsymmetric,
                            partition=:row)

order = LargestFirst()

algo = GreedyColoringAlgorithm(order;
                               decompression=:direct)

result = coloring(S, problem, algo)
```

```@example tutorial
rcolors = row_colors(result)
rgroups = row_groups(result)
num_colors = ncolors(result)
```

```@example tutorial
A_img, B_img = show_colors(result, scale=5, pad=2) # hide
A_img # hide
```

## Star bicoloring

```@example tutorial; continued = true
using SparseMatrixColorings, MatrixMarket

S = MatrixMarket.mmread("smc.mtx")

problem = ColoringProblem(; structure=:nonsymmetric,
                            partition=:bidirectional)

order = RandomOrder()

algo = GreedyColoringAlgorithm(order;
                               decompression=:direct,
                               postprocessing=true)

result = coloring(S, problem, algo)
```

```@example tutorial
rcolors = row_colors(result)
rgroups = row_groups(result)
ccolors = column_colors(result)
cgroups = column_groups(result)
num_colors = ncolors(result)
```

```@example tutorial
A_img, Br_img, Bc_img = show_colors(result, scale=5, pad=2) # hide
A_img # hide
```

## Acyclic bicoloring

```@example tutorial; continued = true
problem = ColoringProblem(; structure=:nonsymmetric,
                            partition=:bidirectional)

order = RandomOrder()

algo = GreedyColoringAlgorithm(order;
                               decompression=:substitution,
                               postprocessing=true)

result = coloring(S, problem, algo)
```

```@example tutorial
rcolors = row_colors(result)
rgroups = row_groups(result)
ccolors = column_colors(result)
cgroups = column_groups(result)
num_colors = ncolors(result)
```

```@example tutorial
A_img, Br_img, Bc_img = show_colors(result, scale=5, pad=2) # hide
A_img # hide
```

## Symmetric star coloring

```@example tutorial; continued = true
problem = ColoringProblem(; structure=:symmetric,
                            partition=:column)

order = NaturalOrder()

algo = GreedyColoringAlgorithm(order;
                               decompression=:direct,
                               postprocessing=false)

result = coloring(S, problem, algo)
```

```@example tutorial
ccolors = column_colors(result)
cgroups = column_groups(result)
num_colors = ncolors(result)
```

```@example tutorial
A_img, B_img = show_colors(result, scale=5, pad=2) # hide
A_img # hide
```

## Symmetric acyclic coloring

```@example tutorial; continued = true
problem = ColoringProblem(; structure=:symmetric,
                            partition=:column)

order = NaturalOrder()

algo = GreedyColoringAlgorithm(order;
                               decompression=:substitution,
                               postprocessing=false)

result = coloring(S, problem, algo)
```

```@example tutorial
ccolors = column_colors(result)
cgroups = column_groups(result)
num_colors = ncolors(result)
```

```@example tutorial
A_img, B_img = show_colors(result, scale=5, pad=2) # hide
A_img # hide
```
