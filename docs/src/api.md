# API reference

```@meta
CollapsedDocStrings = true
CurrentModule = SparseMatrixColorings
```

## Public, exported

```@docs
SparseMatrixColorings
GreedyColoringAlgorithm
column_coloring
row_coloring
symmetric_coloring
symmetric_coloring_detailed
```

## Public, not exported

### Orders

```@docs
AbstractOrder
NaturalOrder
RandomOrder
LargestFirst
```

### Decompression

```@docs
color_groups
decompress_columns
decompress_columns!
decompress_rows
decompress_rows!
decompress_symmetric
decompress_symmetric!
StarSet
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

### Coloring

```@docs
partial_distance2_coloring
star_coloring
star_coloring_detailed
```

### Testing

```@docs
structurally_orthogonal_columns
symmetrically_orthogonal_columns
directly_recoverable_columns
```
