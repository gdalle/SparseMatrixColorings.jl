# SparseMatrixColorings.jl

[![Build Status](https://github.com/gdalle/SparseMatrixColorings.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/gdalle/SparseMatrixColorings.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://gdalle.github.io/SparseMatrixColorings.jl/stable/)
[![Dev Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://gdalle.github.io/SparseMatrixColorings.jl/dev/)
[![Coverage](https://codecov.io/gh/gdalle/SparseMatrixColorings.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/gdalle/SparseMatrixColorings.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)
[![arXiv](https://img.shields.io/badge/arXiv-2505.07308-b31b1b.svg)](https://arxiv.org/abs/2505.07308)
[![DOI](https://zenodo.org/badge/801999408.svg)](https://zenodo.org/doi/10.5281/zenodo.11314275)

Coloring algorithms for sparse Jacobian and Hessian matrices.

## Getting started

To install this package, run the following in a Julia Pkg REPL:

```julia
pkg> add SparseMatrixColorings
```

## Background

The algorithms implemented in this package are described in the following preprint:

- [_Revisiting Sparse Matrix Coloring and Bicoloring_](https://arxiv.org/abs/2505.07308), Montoison et al. (2025)

and inspired by previous works:

- [_What Color Is Your Jacobian? Graph Coloring for Computing Derivatives_](https://epubs.siam.org/doi/10.1137/S0036144504444711), Gebremedhin et al. (2005)
- [_New Acyclic and Star Coloring Algorithms with Application to Computing Hessians_](https://epubs.siam.org/doi/abs/10.1137/050639879), Gebremedhin et al. (2007)
- [_Efficient Computation of Sparse Hessians Using Coloring and Automatic Differentiation_](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.1080.0286), Gebremedhin et al. (2009)
- [_ColPack: Software for graph coloring and related problems in scientific computing_](https://dl.acm.org/doi/10.1145/2513109.2513110), Gebremedhin et al. (2013)

Some parts of the articles (like definitions) are thus copied verbatim in the documentation.

## Alternatives

- [ColPack.jl](https://github.com/michel2323/ColPack.jl): a Julia interface to the C++ library [ColPack](https://github.com/CSCsw/ColPack)
- [SparseDiffTools.jl](https://github.com/JuliaDiff/SparseDiffTools.jl): contains Julia implementations of some coloring algorithms

## Citing

Please cite this software using the provided `CITATION.cff` file or the `.bib` entry below:

```bibtex
@unpublished{montoison2025revisitingsparsematrixcoloring,
      title={Revisiting Sparse Matrix Coloring and Bicoloring}, 
      author={Alexis Montoison and Guillaume Dalle and Assefaw Gebremedhin},
      year={2025},
      eprint={2505.07308},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2505.07308}, 
}
```

The link <https://zenodo.org/doi/10.5281/zenodo.11314275> resolves to the latest version on Zenodo.
