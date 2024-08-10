# API reference

```@meta
CollapsedDocStrings = true
CurrentModule = SparseMatrixColorings
```

```@docs
SparseMatrixColorings
```

The docstrings on this page define the public API of the package.

## Main function

```@docs
coloring
ColoringProblem
GreedyColoringAlgorithm
```

## Result analysis

```@docs
AbstractColoringResult
column_colors
row_colors
column_groups
row_groups
```

## Decompression

```@docs
decompress
decompress!
```

## Orders

These symbols are not exported but they are still part of the public API.

```@docs
AbstractOrder
NaturalOrder
RandomOrder
LargestFirst
```
