# API reference

```@meta
CollapsedDocStrings = true
CurrentModule = SparseMatrixColorings
```

## Public, exported

```@docs
SparseMatrixColorings
GreedyColoringAlgorithm
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
decompress_columns!
decompress_columns
decompress_rows!
decompress_rows
color_groups
```

## Private

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
star_coloring1
```

### Testing

```@docs
check_structurally_orthogonal_columns
check_structurally_orthogonal_rows
check_symmetrically_orthogonal
```
