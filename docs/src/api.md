# API reference

```@meta
CollapsedDocStrings = true
CurrentModule = SparseMatrixColorings
```

## Public, exported

```@docs
SparseMatrixColorings
GreedyColoringAlgorithm
column_coloring_detailed
row_coloring_detailed
symmetric_coloring_detailed
get_colors
get_groups
```

## Public, not exported

### Coloring

```@docs
AbstractColoringResult
```

### Orders

```@docs
AbstractOrder
NaturalOrder
RandomOrder
LargestFirst
```

### Decompression

```@docs
decompress_columns
decompress_columns!
decompress_rows
decompress_rows!
decompress_symmetric
decompress_symmetric!
```

## Private

### Matrices

```@docs
matrix_versions
respectful_similar
same_sparsity_pattern
```

### Graphs

```@docs
Graph
BipartiteGraph
adjacency_graph
bipartite_graph
neighbors
vertices
```

### Coloring (low level)

```@docs
group_by_color
partial_distance2_coloring
star_coloring
symmetric_coefficient
StarSet
```

### Testing

```@docs
structurally_orthogonal_columns
symmetrically_orthogonal_columns
directly_recoverable_columns
```
