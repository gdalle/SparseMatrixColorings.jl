# Dev docs

```@meta
CollapsedDocStrings = true
CurrentModule = SparseMatrixColorings
```

The docstrings on this page describe internals, they are not part of the public API.

## Graph storage

```@docs
SparseMatrixColorings.Graph
SparseMatrixColorings.BipartiteGraph
SparseMatrixColorings.vertices
SparseMatrixColorings.neighbors
SparseMatrixColorings.adjacency_graph
SparseMatrixColorings.bipartite_graph
```

## Low-level coloring

```@docs
SparseMatrixColorings.partial_distance2_coloring
SparseMatrixColorings.symmetric_coefficient
SparseMatrixColorings.star_coloring
SparseMatrixColorings.acyclic_coloring
SparseMatrixColorings.group_by_color
SparseMatrixColorings.get_matrix
SparseMatrixColorings.StarSet
SparseMatrixColorings.TreeSet
```

## Concrete coloring results

```@docs
SparseMatrixColorings.DefaultColoringResult
SparseMatrixColorings.DirectSparseColoringResult
```

## Testing

```@docs
SparseMatrixColorings.same_sparsity_pattern
SparseMatrixColorings.directly_recoverable_columns
SparseMatrixColorings.symmetrically_orthogonal_columns
SparseMatrixColorings.structurally_orthogonal_columns
```

## Matrix handling

```@docs
SparseMatrixColorings.respectful_similar
SparseMatrixColorings.matrix_versions
```

## Examples

```@docs
SparseMatrixColorings.Example
SparseMatrixColorings.what_fig_41
SparseMatrixColorings.what_fig_61
SparseMatrixColorings.efficient_fig_1
SparseMatrixColorings.efficient_fig_4
```
