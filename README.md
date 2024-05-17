# SparseMatrixColorings.jl

[![Build Status](https://github.com/gdalle/SparseMatrixColorings.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/gdalle/SparseMatrixColorings.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Dev Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://gdalle.github.io/SparseMatrixColorings.jl/dev/)
[![Coverage](https://codecov.io/gh/gdalle/SparseMatrixColorings.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/gdalle/SparseMatrixColorings.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

Coloring algorithms for sparse Jacobian and Hessian matrices.

## Getting started

To install this package, run the following in a Julia Pkg REPL:

```julia
pkg> add https://github.com/gdalle/SparseMatrixColorings.jl
```

## Background

The algorithms implemented in this package are all taken from the following survey:

> [_What Color Is Your Jacobian? Graph Coloring for Computing Derivatives_](https://epubs.siam.org/doi/10.1137/S0036144504444711), Gebremedhin et al. (2005)

Some parts of the survey (like definitions and theorems) are also copied verbatim or referred to by their number in the documentation.

## Alternatives

- [ColPack.jl](https://github.com/michel2323/ColPack.jl): a Julia interface to the C++ library [ColPack](https://github.com/CSCsw/ColPack)
- [SparseDiffTools.jl](https://github.com/JuliaDiff/SparseDiffTools.jl): contains Julia implementations of some coloring algorithms
