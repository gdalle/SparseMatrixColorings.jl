# API reference

```@meta
CollapsedDocStrings = true
CurrentModule = SparseMatrixColorings
```

## Public, exported

```@docs
SparseMatrixColorings
```

### Entry point

```@docs
coloring
ColoringProblem
GreedyColoringAlgorithm
```

### Result analysis

```@docs
AbstractColoringResult
column_colors
row_colors
column_groups
row_groups
```

### ADTypes interface

```@docs
column_coloring
row_coloring
symmetric_coloring
```

## Public, not exported

### Decompression

```@docs
decompress
decompress!
```

### Orders

```@docs
AbstractOrder
NaturalOrder
RandomOrder
LargestFirst
```

## Private

### Graph storage

```@docs
SparseMatrixColorings.Graph
SparseMatrixColorings.BipartiteGraph
SparseMatrixColorings.vertices
SparseMatrixColorings.neighbors
SparseMatrixColorings.adjacency_graph
SparseMatrixColorings.bipartite_graph
```

### Low-level coloring

```@docs
SparseMatrixColorings.partial_distance2_coloring
SparseMatrixColorings.symmetric_coefficient
SparseMatrixColorings.star_coloring
SparseMatrixColorings.StarSet
SparseMatrixColorings.group_by_color
SparseMatrixColorings.get_matrix
```

### Concrete coloring results

```@docs
SparseMatrixColorings.DefaultColoringResult
SparseMatrixColorings.DirectSparseColoringResult
```

### Testing

```@docs
SparseMatrixColorings.same_sparsity_pattern
SparseMatrixColorings.directly_recoverable_columns
SparseMatrixColorings.symmetrically_orthogonal_columns
SparseMatrixColorings.structurally_orthogonal_columns
```

### Matrix handling

```@docs
SparseMatrixColorings.respectful_similar
SparseMatrixColorings.matrix_versions
```

### Examples

```@docs
SparseMatrixColorings.Example
SparseMatrixColorings.what_fig_41
SparseMatrixColorings.efficient_fig_1
```
