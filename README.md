# SparseMatrixColorings.jl

[![Build Status](https://github.com/gdalle/SparseMatrixColorings.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/gdalle/SparseMatrixColorings.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://gdalle.github.io/SparseMatrixColorings.jl/stable/)
[![Dev Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://gdalle.github.io/SparseMatrixColorings.jl/dev/)
[![Coverage](https://codecov.io/gh/gdalle/SparseMatrixColorings.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/gdalle/SparseMatrixColorings.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

Coloring algorithms for sparse Jacobian and Hessian matrices.

## Getting started

To install this package, run the following in a Julia Pkg REPL:

```julia
pkg> add SparseMatrixColorings
```

## Background

The algorithms implemented in this package are taken from the following articles:

- [_What Color Is Your Jacobian? Graph Coloring for Computing Derivatives_](https://epubs.siam.org/doi/10.1137/S0036144504444711), Gebremedhin et al. (2005)
- [_New Acyclic and Star Coloring Algorithms with Application to Computing Hessians_](https://epubs.siam.org/doi/abs/10.1137/050639879), Gebremedhin et al. (2007)
- [_Efficient Computation of Sparse Hessians Using Coloring and Automatic Differentiation_](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.1080.0286), Gebremedhin et al. (2009)
- [_ColPack: Software for graph coloring and related problems in scientific computing_](https://dl.acm.org/doi/10.1145/2513109.2513110), Gebremedhin et al. (2013)

Some parts of the articles (like definitions) are thus copied verbatim in the documentation.

## Alternatives

- [ColPack.jl](https://github.com/michel2323/ColPack.jl): a Julia interface to the C++ library [ColPack](https://github.com/CSCsw/ColPack)
- [SparseDiffTools.jl](https://github.com/JuliaDiff/SparseDiffTools.jl): contains Julia implementations of some coloring algorithms
