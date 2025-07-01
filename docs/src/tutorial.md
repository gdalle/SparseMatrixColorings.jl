# Tutorial

Here we give a brief introduction to the contents of the package, see the [API reference](@ref) for more details.

```@example tutorial
using SparseMatrixColorings
using LinearAlgebra
using SparseArrays
using StableRNGs
using SparseMatrixColorings: show_colors # hide
using Images # hide

scale=15 # hide
pad=3 # hide
border=2 # hide
nothing # hide
```

## Coloring problems and algorithms

SparseMatrixColorings.jl is based on the combination of a coloring problem and a coloring algorithm, which will be passed to the [`coloring`](@ref) function.

The problem defines what you want to solve. It is always a [`ColoringProblem`](@ref), and you can select options such as

- the structure of the matrix (`:nonsymmetric` or `:symmetric`)
- the type of partition you want (`:column`, `:row` or `:bidirectional`).

```@example tutorial
problem = ColoringProblem()
```

The algorithm defines how you want to solve it. It can be either a [`GreedyColoringAlgorithm`](@ref) or a [`ConstantColoringAlgorithm`](@ref). For `GreedyColoringAlgorithm`, you can select options such as

- the order in which vertices are processed (a subtype of [`AbstractOrder`](@ref SparseMatrixColorings.AbstractOrder))
- the type of decompression you want (`:direct` or `:substitution`)

```@example tutorial
algo = GreedyColoringAlgorithm()
```

## Coloring results

The [`coloring`](@ref) function takes a matrix, a problem and an algorithm, to return a result subtyping [`AbstractColoringResult`](@ref).

```@example tutorial
S = sparse([
    0 0 1 1 0 1
    1 0 0 0 1 0
    0 1 0 0 1 0
    0 1 1 0 0 0
])

result = coloring(S, problem, algo)
```

The detailed type and fields of that result are _not part of the public API_.
To access its contents, you can use the following getters:

- [`sparsity_pattern`](@ref) for the matrix initially provided to `coloring`
- [`ncolors`](@ref) for the total number of distinct colors
- [`row_colors`](@ref), [`column_colors`](@ref) for vectors of integer colors (depending on the partition)
- [`row_groups`](@ref), [`column_groups`](@ref) for vector of row or column indices grouped by color (depending on the partition)

Here, we have a column coloring, so we can try the following:

```@example tutorial
column_colors(result)
```

```@example tutorial
column_groups(result)
```

```@example tutorial
ncolors(result)
```

## Compression and decompression

The functions [`compress`](@ref) and [`decompress`](@ref) efficiently store and retrieve compressed representations of sparse matrices, using the coloring result as a starting point.

Compression sums all columns or rows with the same color:

```@example tutorial
M = sparse([
    0 0 4 6 0 9
    1 0 0 0 7 0
    0 2 0 0 8 0
    0 3 5 0 0 0
])

B = compress(M, result)
```

Decompression recovers the original matrix from its compressed version:

```@example tutorial
C = decompress(B, result)
```

The functions [`decompress!`](@ref) and [`decompress_single_color!`](@ref) are in-place variants of [`decompress`](@ref).

```@example tutorial
D = [
    10 14 18
    11 15 0
    12 16 0
    13 17 0
]

decompress!(C, D, result)
```

```@example tutorial
nonzeros(C) .= -1
decompress_single_color!(C, D[:, 2], 2, result)
```

## Unidirectional variants

We now illustrate the variants of colorings available, on the following matrix:

```@example tutorial
S = sparse(Symmetric(sprand(StableRNG(0), Bool, 10, 10, 0.4)))
```

We start with unidirectional colorings, where only rows or columns are colored and the matrix is not assumed to be symmetric.

### Column coloring

```@example tutorial
problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)
algo = GreedyColoringAlgorithm(; decompression=:direct)
result = coloring(S, problem, algo)
nothing # hide
```

```@example tutorial
ncolors(result), column_colors(result)
```

Here is the colored matrix

```@example tutorial
A_img, B_img = show_colors(result; scale, pad, border) # hide
A_img # hide
```

and its columnwise compression

```@example tutorial
B_img # hide
```

### Row coloring

```@example tutorial
problem = ColoringProblem(; structure=:nonsymmetric, partition=:row)
algo = GreedyColoringAlgorithm(; decompression=:direct)
result = coloring(S, problem, algo)
nothing # hide
```

```@example tutorial
ncolors(result), row_colors(result)
```

Here is the colored matrix

```@example tutorial
A_img, B_img = show_colors(result; scale, pad, border) # hide
A_img # hide
```

and its rowwise compression

```@example tutorial
B_img # hide
```

## Symmetric variants

We continue with unidirectional symmetric colorings, where coloring rows is equivalent to coloring columns.
Symmetry is leveraged to possibly reduce the number of necessary colors.

### Star coloring

Star coloring is the algorithm used for symmetric matrices with direct decompression.

```@example tutorial
problem = ColoringProblem(; structure=:symmetric, partition=:column)
algo = GreedyColoringAlgorithm(; decompression=:direct)
result = coloring(S, problem, algo)
nothing # hide
```

```@example tutorial
ncolors(result), column_colors(result)
```

Here is the colored matrix

```@example tutorial
A_img, B_img = show_colors(result; scale, pad, border) # hide
A_img # hide
```

and its columnwise compression

```@example tutorial
B_img # hide
```

### Acyclic coloring

Acyclic coloring is the algorithm used for symmetric matrices with decompression by substitution.

```@example tutorial
problem = ColoringProblem(; structure=:symmetric, partition=:column)
algo = GreedyColoringAlgorithm(; decompression=:substitution)
result = coloring(S, problem, algo)
nothing # hide
```

```@example tutorial
ncolors(result), column_colors(result)
```

Here is the colored matrix

```@example tutorial
A_img, B_img = show_colors(result; scale, pad, border) # hide
A_img # hide
```

and its columnwise compression

```@example tutorial
B_img # hide
```

## Bidirectional variants

We finish with bidirectional colorings, where both rows and columns are colored and the matrix is not assumed to be symmetric.

Bicoloring is most relevant for matrices with dense rows and columns, which is why we consider the following test case:

```@example tutorial
S_bi = copy(S)
S_bi[:, 1] .= true
S_bi[1, :] .= true
S_bi
```

With our implementations, bidirectional coloring works better using a [`RandomOrder`](@ref).

### Star bicoloring

```@example tutorial
problem = ColoringProblem(; structure=:nonsymmetric, partition=:bidirectional)
algo = GreedyColoringAlgorithm(RandomOrder(StableRNG(0), 0); decompression=:direct)
result = coloring(S_bi, problem, algo)
nothing # hide
```

```@example tutorial
ncolors(result), column_colors(result)
```

Here is the colored matrix

```@example tutorial
Arc_img, _, _, Br_img, Bc_img = show_colors(result; scale, pad, border) # hide
Arc_img # hide
```

its columnwise compression

```@example tutorial
Bc_img # hide
```

and rowwise compression

```@example tutorial
Br_img # hide
```

Both are necessary to reconstruct the original.

#### Acyclic bicoloring

```@example tutorial
problem = ColoringProblem(; structure=:nonsymmetric, partition=:bidirectional)
algo = GreedyColoringAlgorithm(RandomOrder(StableRNG(0), 0); decompression=:substitution)
result = coloring(S_bi, problem, algo)
nothing # hide
```

```@example tutorial
ncolors(result), column_colors(result)
```

Here is the colored matrix

```@example tutorial
Arc_img, _, _, Br_img, Bc_img = show_colors(result; scale=20, pad, border) # hide
Arc_img # hide
```

its columnwise compression

```@example tutorial
Bc_img # hide
```

and rowwise compression

```@example tutorial
Br_img # hide
```

Both are necessary to reconstruct the original.
