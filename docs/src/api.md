# API reference

```@meta
CollapsedDocStrings = true
CurrentModule = SparseMatrixColorings
```

The docstrings on this page define the public API of the package.

```@docs
SparseMatrixColorings
```

## Main function

```@docs
coloring
fast_coloring
ColoringProblem
GreedyColoringAlgorithm
ConstantColoringAlgorithm
```

## Result analysis

```@docs
AbstractColoringResult
column_colors
row_colors
ncolors
column_groups
row_groups
sparsity_pattern
```

## Decompression

```@docs
compress
decompress
decompress!
decompress_single_color!
```

## Orders

```@docs
AbstractOrder
NaturalOrder
RandomOrder
LargestFirst
SmallestLast
IncidenceDegree
DynamicLargestFirst
DynamicDegreeBasedOrder
PerfectEliminationOrder
```
