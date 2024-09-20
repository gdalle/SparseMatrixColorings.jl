var documenterSearchIndex = {"docs":
[{"location":"api/#API-reference","page":"API reference","title":"API reference","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"CollapsedDocStrings = true\nCurrentModule = SparseMatrixColorings","category":"page"},{"location":"api/","page":"API reference","title":"API reference","text":"The docstrings on this page define the public API of the package.","category":"page"},{"location":"api/","page":"API reference","title":"API reference","text":"SparseMatrixColorings","category":"page"},{"location":"api/#SparseMatrixColorings.SparseMatrixColorings","page":"API reference","title":"SparseMatrixColorings.SparseMatrixColorings","text":"SparseMatrixColorings\n\nSparseMatrixColorings.jl\n\n(Image: Build Status) (Image: Stable Documentation) (Image: Dev Documentation) (Image: Coverage) (Image: Code Style: Blue) (Image: DOI)\n\nColoring algorithms for sparse Jacobian and Hessian matrices.\n\nGetting started\n\nTo install this package, run the following in a Julia Pkg REPL:\n\npkg> add SparseMatrixColorings\n\nBackground\n\nThe algorithms implemented in this package are taken from the following articles:\n\nWhat Color Is Your Jacobian? Graph Coloring for Computing Derivatives, Gebremedhin et al. (2005)\nNew Acyclic and Star Coloring Algorithms with Application to Computing Hessians, Gebremedhin et al. (2007)\nEfficient Computation of Sparse Hessians Using Coloring and Automatic Differentiation, Gebremedhin et al. (2009)\nColPack: Software for graph coloring and related problems in scientific computing, Gebremedhin et al. (2013)\n\nSome parts of the articles (like definitions) are thus copied verbatim in the documentation.\n\nAlternatives\n\nColPack.jl: a Julia interface to the C++ library ColPack\nSparseDiffTools.jl: contains Julia implementations of some coloring algorithms\n\nCiting\n\nPlease cite this software using the Zenodo DOI of the version you used. The link https://zenodo.org/doi/10.5281/zenodo.11314275 resolves to the latest version.\n\nExports\n\nAbstractColoringResult\nColoringProblem\nGreedyColoringAlgorithm\nLargestFirst\nNaturalOrder\nRandomOrder\ncoloring\ncolumn_colors\ncolumn_groups\ncompress\ndecompress\ndecompress!\ndecompress_single_color!\nrow_colors\nrow_groups\n\n\n\n\n\n","category":"module"},{"location":"api/#Main-function","page":"API reference","title":"Main function","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"coloring\nColoringProblem\nGreedyColoringAlgorithm","category":"page"},{"location":"api/#SparseMatrixColorings.coloring","page":"API reference","title":"SparseMatrixColorings.coloring","text":"coloring(\n    S::AbstractMatrix,\n    problem::ColoringProblem,\n    algo::GreedyColoringAlgorithm;\n    [decompression_eltype=Float64, symmetric_pattern=false]\n)\n\nSolve a ColoringProblem on the matrix S with a GreedyColoringAlgorithm and return an AbstractColoringResult.\n\nThe result can be used to compress and decompress a matrix A with the same sparsity pattern as S. If eltype(A) == decompression_eltype, decompression might be faster.\n\nFor a :nonsymmetric problem (and only then), setting symmetric_pattern=true indicates that the pattern of nonzeros is symmetric. This condition is weaker than the symmetry of actual values, so it can happen for some Jacobians. Specifying it allows faster construction of the bipartite graph.\n\nExample\n\njulia> using SparseMatrixColorings, SparseArrays\n\njulia> S = sparse([\n           0 0 1 1 0 1\n           1 0 0 0 1 0\n           0 1 0 0 1 0\n           0 1 1 0 0 0\n       ]);\n\njulia> problem = ColoringProblem(; structure=:nonsymmetric, partition=:column);\n\njulia> algo = GreedyColoringAlgorithm(; decompression=:direct);\n\njulia> result = coloring(S, problem, algo);\n\njulia> column_colors(result)\n6-element Vector{Int64}:\n 1\n 1\n 2\n 1\n 2\n 3\n\njulia> column_groups(result)\n3-element Vector{Vector{Int64}}:\n [1, 2, 4]\n [3, 5]\n [6]\n\nSee also\n\nColoringProblem\nGreedyColoringAlgorithm\nAbstractColoringResult\ncompress\ndecompress\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseMatrixColorings.ColoringProblem","page":"API reference","title":"SparseMatrixColorings.ColoringProblem","text":"ColoringProblem{structure,partition}\n\nSelector type for the coloring problem to solve, enabling multiple dispatch.\n\nIt is passed as an argument to the main function coloring.\n\nConstructors\n\nColoringProblem{structure,partition}()\nColoringProblem(; structure=:nonsymmetric, partition=:column)\n\nstructure::Symbol: either :nonsymmetric or :symmetric\npartition::Symbol: either :column, :row or :bidirectional\n\nwarning: Warning\nThe second constructor (based on keyword arguments) is type-unstable.\n\nLink to automatic differentiation\n\nMatrix coloring is often used in automatic differentiation, and here is the translation guide:\n\nmatrix mode structure partition implemented\nJacobian forward :nonsymmetric :column yes\nJacobian reverse :nonsymmetric :row yes\nJacobian mixed :nonsymmetric :bidirectional no\nHessian - :symmetric :column yes\nHessian - :symmetric :row no\n\n\n\n\n\n","category":"type"},{"location":"api/#SparseMatrixColorings.GreedyColoringAlgorithm","page":"API reference","title":"SparseMatrixColorings.GreedyColoringAlgorithm","text":"GreedyColoringAlgorithm{decompression} <: ADTypes.AbstractColoringAlgorithm\n\nGreedy coloring algorithm for sparse matrices which colors columns or rows one after the other, following a configurable order.\n\nIt is passed as an argument to the main function coloring.\n\nConstructors\n\nGreedyColoringAlgorithm{decompression}(order=NaturalOrder())\nGreedyColoringAlgorithm(order=NaturalOrder(); decompression=:direct)\n\norder::AbstractOrder: the order in which the columns or rows are colored, which can impact the number of colors.\ndecompression::Symbol: either :direct or :substitution. Usually :substitution leads to fewer colors, at the cost of a more expensive coloring (and decompression). When :substitution is not applicable, it falls back on :direct decompression.\n\nwarning: Warning\nThe second constructor (based on keyword arguments) is type-unstable.\n\nADTypes coloring interface\n\nGreedyColoringAlgorithm is a subtype of ADTypes.AbstractColoringAlgorithm, which means the following methods are also applicable:\n\nADTypes.column_coloring\nADTypes.row_coloring\nADTypes.symmetric_coloring\n\nSee their respective docstrings for details.\n\nSee also\n\nAbstractOrder\ndecompress\n\n\n\n\n\n","category":"type"},{"location":"api/#Result-analysis","page":"API reference","title":"Result analysis","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"AbstractColoringResult\ncolumn_colors\nrow_colors\ncolumn_groups\nrow_groups","category":"page"},{"location":"api/#SparseMatrixColorings.AbstractColoringResult","page":"API reference","title":"SparseMatrixColorings.AbstractColoringResult","text":"AbstractColoringResult{structure,partition,decompression}\n\nAbstract type for the result of a coloring algorithm.\n\nIt is the supertype of the object returned by the main function coloring.\n\nType parameters\n\nCombination between the type parameters of ColoringProblem and GreedyColoringAlgorithm:\n\nstructure::Symbol: either :nonsymmetric or :symmetric\npartition::Symbol: either :column, :row or :bidirectional\ndecompression::Symbol: either :direct or :substitution\n\nApplicable methods\n\ncolumn_colors and column_groups (for a :column or :bidirectional partition) \nrow_colors and row_groups (for a :row or :bidirectional partition)\ncompress, decompress, decompress!, decompress_single_color!\n\nwarning: Warning\nUnlike the methods above, the concrete subtypes of AbstractColoringResult are not part of the public API and may change without notice.\n\n\n\n\n\n","category":"type"},{"location":"api/#SparseMatrixColorings.column_colors","page":"API reference","title":"SparseMatrixColorings.column_colors","text":"column_colors(result::AbstractColoringResult)\n\nReturn a vector color of integer colors, one for each column of the colored matrix.\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseMatrixColorings.row_colors","page":"API reference","title":"SparseMatrixColorings.row_colors","text":"row_colors(result::AbstractColoringResult)\n\nReturn a vector color of integer colors, one for each row of the colored matrix.\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseMatrixColorings.column_groups","page":"API reference","title":"SparseMatrixColorings.column_groups","text":"column_groups(result::AbstractColoringResult)\n\nReturn a vector group such that for every color c, group[c] contains the indices of all columns that are colored with c.\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseMatrixColorings.row_groups","page":"API reference","title":"SparseMatrixColorings.row_groups","text":"row_groups(result::AbstractColoringResult)\n\nReturn a vector group such that for every color c, group[c] contains the indices of all rows that are colored with c.\n\n\n\n\n\n","category":"function"},{"location":"api/#Decompression","page":"API reference","title":"Decompression","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"compress\ndecompress\ndecompress!\ndecompress_single_color!","category":"page"},{"location":"api/#SparseMatrixColorings.compress","page":"API reference","title":"SparseMatrixColorings.compress","text":"compress(A, result::AbstractColoringResult)\n\nCompress A given a coloring result of the sparsity pattern of A.\n\nIf result comes from a :column (resp. :row) partition, the output is a single matrix B compressed by column (resp. by row).\nIf result comes from a :bidirectional partition, the output is a tuple of matrices (Br, Bc), where Br is compressed by row and Bc by column.\n\nCompression means summing either the columns or the rows of A which share the same color. It is undone by calling decompress or decompress!.\n\nwarning: Warning\nAt the moment, :bidirectional partitions are not implemented.\n\nExample\n\njulia> using SparseMatrixColorings, SparseArrays\n\njulia> A = sparse([\n           0 0 4 6 0 9\n           1 0 0 0 7 0\n           0 2 0 0 8 0\n           0 3 5 0 0 0\n       ]);\n\njulia> result = coloring(A, ColoringProblem(), GreedyColoringAlgorithm());\n\njulia> column_groups(result)\n3-element Vector{Vector{Int64}}:\n [1, 2, 4]\n [3, 5]\n [6]\n\njulia> B = compress(A, result)\n4×3 Matrix{Int64}:\n 6  4  9\n 1  7  0\n 2  8  0\n 3  5  0\n\nSee also\n\nColoringProblem\nAbstractColoringResult\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseMatrixColorings.decompress","page":"API reference","title":"SparseMatrixColorings.decompress","text":"decompress(B::AbstractMatrix, result::AbstractColoringResult)\n\nDecompress B into a new matrix A, given a coloring result of the sparsity pattern of A. The in-place alternative is decompress!.\n\nCompression means summing either the columns or the rows of A which share the same color. It is done by calling compress.\n\nExample\n\njulia> using SparseMatrixColorings, SparseArrays\n\njulia> A = sparse([\n           0 0 4 6 0 9\n           1 0 0 0 7 0\n           0 2 0 0 8 0\n           0 3 5 0 0 0\n       ]);\n\njulia> result = coloring(A, ColoringProblem(), GreedyColoringAlgorithm());\n\njulia> column_groups(result)\n3-element Vector{Vector{Int64}}:\n [1, 2, 4]\n [3, 5]\n [6]\n\njulia> B = compress(A, result)\n4×3 Matrix{Int64}:\n 6  4  9\n 1  7  0\n 2  8  0\n 3  5  0\n\njulia> decompress(B, result)\n4×6 SparseMatrixCSC{Int64, Int64} with 9 stored entries:\n ⋅  ⋅  4  6  ⋅  9\n 1  ⋅  ⋅  ⋅  7  ⋅\n ⋅  2  ⋅  ⋅  8  ⋅\n ⋅  3  5  ⋅  ⋅  ⋅\n\njulia> decompress(B, result) == A\ntrue\n\nSee also\n\nColoringProblem\nAbstractColoringResult\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseMatrixColorings.decompress!","page":"API reference","title":"SparseMatrixColorings.decompress!","text":"decompress!(\n    A::AbstractMatrix, B::AbstractMatrix,\n    result::AbstractColoringResult, [uplo=:F]\n)\n\nDecompress B in-place into A, given a coloring result of the sparsity pattern of A. The out-of-place alternative is decompress.\n\nnote: Note\nIn-place decompression is faster when A isa SparseMatrixCSC.\n\nCompression means summing either the columns or the rows of A which share the same color. It is done by calling compress.\n\nFor :symmetric coloring results (and for those only), an optional positional argument uplo in (:U, :L, :F) can be passed to specify which part of the matrix A should be updated: the Upper triangle, the Lower triangle, or the Full matrix. When A isa SparseMatrixCSC, using the uplo argument requires a target matrix which only stores the relevant triangle(s).\n\nExample\n\njulia> using SparseMatrixColorings, SparseArrays\n\njulia> A = sparse([\n           0 0 4 6 0 9\n           1 0 0 0 7 0\n           0 2 0 0 8 0\n           0 3 5 0 0 0\n       ]);\n\njulia> result = coloring(A, ColoringProblem(), GreedyColoringAlgorithm());\n\njulia> column_groups(result)\n3-element Vector{Vector{Int64}}:\n [1, 2, 4]\n [3, 5]\n [6]\n\njulia> B = compress(A, result)\n4×3 Matrix{Int64}:\n 6  4  9\n 1  7  0\n 2  8  0\n 3  5  0\n\njulia> A2 = similar(A);\n\njulia> decompress!(A2, B, result)\n4×6 SparseMatrixCSC{Int64, Int64} with 9 stored entries:\n ⋅  ⋅  4  6  ⋅  9\n 1  ⋅  ⋅  ⋅  7  ⋅\n ⋅  2  ⋅  ⋅  8  ⋅\n ⋅  3  5  ⋅  ⋅  ⋅\n\njulia> A2 == A\ntrue\n\nSee also\n\nColoringProblem\nAbstractColoringResult\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseMatrixColorings.decompress_single_color!","page":"API reference","title":"SparseMatrixColorings.decompress_single_color!","text":"decompress_single_color!(\n    A::AbstractMatrix, b::AbstractVector, c::Integer,\n    result::AbstractColoringResult, [uplo=:F]\n)\n\nDecompress the vector b corresponding to color c in-place into A, given a :direct coloring result of the sparsity pattern of A (it will not work with a :substitution coloring).\n\nIf result comes from a :nonsymmetric structure with :column partition, this will update the columns of A that share color c (whose sum makes up b).\nIf result comes from a :nonsymmetric structure with :row partition, this will update the rows of A that share color c (whose sum makes up b).\nIf result comes from a :symmetric structure with :column partition, this will update the coefficients of A whose value is deduced from color c.\n\nwarning: Warning\nThis function will only update some coefficients of A, without resetting the rest to zero.\n\nFor :symmetric coloring results (and for those only), an optional positional argument uplo in (:U, :L, :F) can be passed to specify which part of the matrix A should be updated: the Upper triangle, the Lower triangle, or the Full matrix. When A isa SparseMatrixCSC, using the uplo argument requires a target matrix which only stores the relevant triangle(s).\n\nExample\n\njulia> using SparseMatrixColorings, SparseArrays\n\njulia> A = sparse([\n           0 0 4 6 0 9\n           1 0 0 0 7 0\n           0 2 0 0 8 0\n           0 3 5 0 0 0\n       ]);\n\njulia> result = coloring(A, ColoringProblem(), GreedyColoringAlgorithm());\n\njulia> column_groups(result)\n3-element Vector{Vector{Int64}}:\n [1, 2, 4]\n [3, 5]\n [6]\n\njulia> B = compress(A, result)\n4×3 Matrix{Int64}:\n 6  4  9\n 1  7  0\n 2  8  0\n 3  5  0\n\njulia> A2 = similar(A); A2 .= 0;\n\njulia> decompress_single_color!(A2, B[:, 2], 2, result)\n4×6 SparseMatrixCSC{Int64, Int64} with 9 stored entries:\n ⋅  ⋅  4  0  ⋅  0\n 0  ⋅  ⋅  ⋅  7  ⋅\n ⋅  0  ⋅  ⋅  8  ⋅\n ⋅  0  5  ⋅  ⋅  ⋅\n\njulia> A2[:, [3, 5]] == A[:, [3, 5]]\ntrue\n\nSee also\n\nColoringProblem\nAbstractColoringResult\ndecompress!\n\n\n\n\n\n","category":"function"},{"location":"api/#Orders","page":"API reference","title":"Orders","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"AbstractOrder\nNaturalOrder\nRandomOrder\nLargestFirst","category":"page"},{"location":"api/#SparseMatrixColorings.AbstractOrder","page":"API reference","title":"SparseMatrixColorings.AbstractOrder","text":"AbstractOrder\n\nAbstract supertype for the vertex order used inside GreedyColoringAlgorithm.\n\nIn this algorithm, the rows and columns of a matrix form a graph, and the vertices are colored one after the other in a greedy fashion. Depending on how the vertices are ordered, the number of colors necessary may vary.\n\nSubtypes\n\nNaturalOrder\nRandomOrder\nLargestFirst\n\n\n\n\n\n","category":"type"},{"location":"api/#SparseMatrixColorings.NaturalOrder","page":"API reference","title":"SparseMatrixColorings.NaturalOrder","text":"NaturalOrder()\n\nInstance of AbstractOrder which sorts vertices using their index in the provided graph.\n\n\n\n\n\n","category":"type"},{"location":"api/#SparseMatrixColorings.RandomOrder","page":"API reference","title":"SparseMatrixColorings.RandomOrder","text":"RandomOrder(rng=default_rng())\n\nInstance of AbstractOrder which sorts vertices using a random permutation.\n\n\n\n\n\n","category":"type"},{"location":"api/#SparseMatrixColorings.LargestFirst","page":"API reference","title":"SparseMatrixColorings.LargestFirst","text":"LargestFirst()\n\nInstance of AbstractOrder which sorts vertices using their degree in the provided graph: the largest degree comes first.\n\n\n\n\n\n","category":"type"},{"location":"dev/#Dev-docs","page":"Dev docs","title":"Dev docs","text":"","category":"section"},{"location":"dev/","page":"Dev docs","title":"Dev docs","text":"CollapsedDocStrings = true\nCurrentModule = SparseMatrixColorings","category":"page"},{"location":"dev/","page":"Dev docs","title":"Dev docs","text":"The docstrings on this page describe internals, they are not part of the public API.","category":"page"},{"location":"dev/#Graph-storage","page":"Dev docs","title":"Graph storage","text":"","category":"section"},{"location":"dev/","page":"Dev docs","title":"Dev docs","text":"SparseMatrixColorings.Graph\nSparseMatrixColorings.BipartiteGraph\nSparseMatrixColorings.vertices\nSparseMatrixColorings.neighbors\nSparseMatrixColorings.adjacency_graph\nSparseMatrixColorings.bipartite_graph","category":"page"},{"location":"dev/#SparseMatrixColorings.Graph","page":"Dev docs","title":"SparseMatrixColorings.Graph","text":"Graph{T}\n\nUndirected graph structure stored in Compressed Sparse Column (CSC) format.\n\nFields\n\ncolptr::Vector{T}: same as for SparseMatrixCSC\nrowval::Vector{T}: same as for SparseMatrixCSC\n\n\n\n\n\n","category":"type"},{"location":"dev/#SparseMatrixColorings.BipartiteGraph","page":"Dev docs","title":"SparseMatrixColorings.BipartiteGraph","text":"BipartiteGraph{T}\n\nUndirected bipartite graph structure stored in bidirectional Compressed Sparse Column format (redundancy allows for faster access).\n\nA bipartite graph has two \"sides\", which we number 1 and 2.\n\nFields\n\ng1::Graph{T}: contains the neighbors for vertices on side 1\ng2::Graph{T}: contains the neighbors for vertices on side 2\n\n\n\n\n\n","category":"type"},{"location":"dev/#SparseMatrixColorings.vertices","page":"Dev docs","title":"SparseMatrixColorings.vertices","text":"vertices(bg::BipartiteGraph, Val(side))\n\nReturn the list of vertices of bg from the specified side as a range 1:n.\n\n\n\n\n\n","category":"function"},{"location":"dev/#SparseMatrixColorings.neighbors","page":"Dev docs","title":"SparseMatrixColorings.neighbors","text":"neighbors(bg::BipartiteGraph, Val(side), v::Integer)\n\nReturn the neighbors of v (a vertex from the specified side, 1 or 2), in the graph bg.\n\n\n\n\n\n","category":"function"},{"location":"dev/#SparseMatrixColorings.adjacency_graph","page":"Dev docs","title":"SparseMatrixColorings.adjacency_graph","text":"adjacency_graph(A::SparseMatrixCSC)\n\nReturn a Graph representing the nonzeros of a symmetric matrix (typically a Hessian matrix).\n\nThe adjacency graph of a symmetrix matric A ∈ ℝ^{n × n} is G(A) = (V, E) where\n\nV = 1:n is the set of rows or columns i/j\n(i, j) ∈ E whenever A[i, j] ≠ 0 and i ≠ j\n\nReferences\n\nWhat Color Is Your Jacobian? Graph Coloring for Computing Derivatives, Gebremedhin et al. (2005)\n\n\n\n\n\n","category":"function"},{"location":"dev/#SparseMatrixColorings.bipartite_graph","page":"Dev docs","title":"SparseMatrixColorings.bipartite_graph","text":"bipartite_graph(A::SparseMatrixCSC; symmetric_pattern::Bool)\n\nReturn a BipartiteGraph representing the nonzeros of a non-symmetric matrix (typically a Jacobian matrix).\n\nThe bipartite graph of a matrix A ∈ ℝ^{m × n} is Gb(A) = (V₁, V₂, E) where\n\nV₁ = 1:m is the set of rows i\nV₂ = 1:n is the set of columns j\n(i, j) ∈ E whenever A[i, j] ≠ 0\n\nWhen symmetric_pattern is true, this construction is more efficient.\n\nReferences\n\nWhat Color Is Your Jacobian? Graph Coloring for Computing Derivatives, Gebremedhin et al. (2005)\n\n\n\n\n\n","category":"function"},{"location":"dev/#Low-level-coloring","page":"Dev docs","title":"Low-level coloring","text":"","category":"section"},{"location":"dev/","page":"Dev docs","title":"Dev docs","text":"SparseMatrixColorings.partial_distance2_coloring\nSparseMatrixColorings.symmetric_coefficient\nSparseMatrixColorings.star_coloring\nSparseMatrixColorings.acyclic_coloring\nSparseMatrixColorings.group_by_color\nSparseMatrixColorings.StarSet\nSparseMatrixColorings.TreeSet","category":"page"},{"location":"dev/#SparseMatrixColorings.partial_distance2_coloring","page":"Dev docs","title":"SparseMatrixColorings.partial_distance2_coloring","text":"partial_distance2_coloring(bg::BipartiteGraph, ::Val{side}, order::AbstractOrder)\n\nCompute a distance-2 coloring of the given side (1 or 2) in the bipartite graph bg and return a vector of integer colors.\n\nA distance-2 coloring is such that two vertices have different colors if they are at distance at most 2.\n\nThe vertices are colored in a greedy fashion, following the order supplied.\n\nSee also\n\nBipartiteGraph\nAbstractOrder\n\nReferences\n\nWhat Color Is Your Jacobian? Graph Coloring for Computing Derivatives, Gebremedhin et al. (2005), Algorithm 3.2\n\n\n\n\n\n","category":"function"},{"location":"dev/#SparseMatrixColorings.symmetric_coefficient","page":"Dev docs","title":"SparseMatrixColorings.symmetric_coefficient","text":"symmetric_coefficient(\n    i::Integer, j::Integer,\n    color::AbstractVector{<:Integer},\n    star_set::StarSet\n)\n\nReturn the indices (k, c) such that A[i, j] = B[k, c], where A is the uncompressed symmetric matrix and B is the column-compressed matrix.\n\nThis function corresponds to algorithm DirectRecover2 in the paper.\n\nReferences\n\nEfficient Computation of Sparse Hessians Using Coloring and Automatic Differentiation, Gebremedhin et al. (2009), Figure 3\n\n\n\n\n\n","category":"function"},{"location":"dev/#SparseMatrixColorings.star_coloring","page":"Dev docs","title":"SparseMatrixColorings.star_coloring","text":"star_coloring(g::Graph, order::AbstractOrder)\n\nCompute a star coloring of all vertices in the adjacency graph g and return a tuple (color, star_set), where\n\ncolor is the vector of integer colors\nstar_set is a StarSet encoding the set of 2-colored stars\n\nA star coloring is a distance-1 coloring such that every path on 4 vertices uses at least 3 colors.\n\nThe vertices are colored in a greedy fashion, following the order supplied.\n\nSee also\n\nGraph\nAbstractOrder\n\nReferences\n\nNew Acyclic and Star Coloring Algorithms with Application to Computing Hessians, Gebremedhin et al. (2007), Algorithm 4.1\n\n\n\n\n\n","category":"function"},{"location":"dev/#SparseMatrixColorings.acyclic_coloring","page":"Dev docs","title":"SparseMatrixColorings.acyclic_coloring","text":"acyclic_coloring(g::Graph, order::AbstractOrder)\n\nCompute an acyclic coloring of all vertices in the adjacency graph g and return a tuple (color, tree_set), where\n\ncolor is the vector of integer colors\ntree_set is a TreeSet encoding the set of 2-colored trees\n\nAn acyclic coloring is a distance-1 coloring with the further restriction that every cycle uses at least 3 colors.\n\nThe vertices are colored in a greedy fashion, following the order supplied.\n\nSee also\n\nGraph\nAbstractOrder\n\nReferences\n\nNew Acyclic and Star Coloring Algorithms with Application to Computing Hessians, Gebremedhin et al. (2007), Algorithm 3.1\n\n\n\n\n\n","category":"function"},{"location":"dev/#SparseMatrixColorings.group_by_color","page":"Dev docs","title":"SparseMatrixColorings.group_by_color","text":"group_by_color(color::Vector{Int})\n\nCreate group::Vector{Vector{Int}} such that i ∈ group[c] iff color[i] == c.\n\nAssumes the colors are contiguously numbered from 1 to some cmax.\n\n\n\n\n\n","category":"function"},{"location":"dev/#SparseMatrixColorings.StarSet","page":"Dev docs","title":"SparseMatrixColorings.StarSet","text":"StarSet\n\nEncode a set of 2-colored stars resulting from the star_coloring algorithm.\n\nFields\n\nstar::Dict{Tuple{Int64, Int64}, Int64}: a mapping from edges (pair of vertices) to their star index\nhub::Vector{Int64}: a mapping from star indices to their hub (undefined hubs for single-edge stars are the negative value of one of the vertices, picked arbitrarily)\nspokes::Vector{Vector{Int64}}: a mapping from star indices to the vector of their spokes\n\n\n\n\n\n","category":"type"},{"location":"dev/#SparseMatrixColorings.TreeSet","page":"Dev docs","title":"SparseMatrixColorings.TreeSet","text":"TreeSet\n\nEncode a set of 2-colored trees resulting from the acyclic_coloring algorithm.\n\nFields\n\nforest::DataStructures.DisjointSets{Tuple{Int64, Int64}}: a forest of two-colored trees\n\n\n\n\n\n","category":"type"},{"location":"dev/#Concrete-coloring-results","page":"Dev docs","title":"Concrete coloring results","text":"","category":"section"},{"location":"dev/","page":"Dev docs","title":"Dev docs","text":"SparseMatrixColorings.ColumnColoringResult\nSparseMatrixColorings.RowColoringResult\nSparseMatrixColorings.StarSetColoringResult\nSparseMatrixColorings.TreeSetColoringResult\nSparseMatrixColorings.LinearSystemColoringResult","category":"page"},{"location":"dev/#SparseMatrixColorings.ColumnColoringResult","page":"Dev docs","title":"SparseMatrixColorings.ColumnColoringResult","text":"struct ColumnColoringResult{M} <: AbstractColoringResult{:nonsymmetric, :column, :direct, M}\n\nStorage for the result of a column coloring with direct decompression.\n\nFields\n\nS::Any: matrix that was colored\ncolor::Vector{Int64}: one integer color for each column or row (depending on partition)\ngroup::Vector{Vector{Int64}}: color groups for columns or rows (depending on partition)\ncompressed_indices::Vector{Int64}: flattened indices mapping the compressed matrix B to the uncompressed matrix A when A isa SparseMatrixCSC. They satisfy nonzeros(A)[k] = vec(B)[compressed_indices[k]]\n\nSee also\n\nAbstractColoringResult\n\n\n\n\n\n","category":"type"},{"location":"dev/#SparseMatrixColorings.RowColoringResult","page":"Dev docs","title":"SparseMatrixColorings.RowColoringResult","text":"struct RowColoringResult{M} <: AbstractColoringResult{:nonsymmetric, :row, :direct, M}\n\nStorage for the result of a row coloring with direct decompression.\n\nFields\n\nSee the docstring of ColumnColoringResult.\n\nS::Any\nSᵀ::Any\ncolor::Vector{Int64}\ngroup::Vector{Vector{Int64}}\ncompressed_indices::Vector{Int64}\n\nSee also\n\nAbstractColoringResult\n\n\n\n\n\n","category":"type"},{"location":"dev/#SparseMatrixColorings.StarSetColoringResult","page":"Dev docs","title":"SparseMatrixColorings.StarSetColoringResult","text":"struct StarSetColoringResult{M} <: AbstractColoringResult{:symmetric, :column, :direct, M}\n\nStorage for the result of a symmetric coloring with direct decompression.\n\nFields\n\nSee the docstring of ColumnColoringResult.\n\nS::Any\ncolor::Vector{Int64}\ngroup::Vector{Vector{Int64}}\nstar_set::SparseMatrixColorings.StarSet\ncompressed_indices::Vector{Int64}\n\nSee also\n\nAbstractColoringResult\n\n\n\n\n\n","category":"type"},{"location":"dev/#SparseMatrixColorings.TreeSetColoringResult","page":"Dev docs","title":"SparseMatrixColorings.TreeSetColoringResult","text":"struct TreeSetColoringResult{M, R} <: AbstractColoringResult{:symmetric, :column, :substitution, M}\n\nStorage for the result of a symmetric coloring with decompression by substitution.\n\nFields\n\nSee the docstring of ColumnColoringResult.\n\nS::Any\ncolor::Vector{Int64}\ngroup::Vector{Vector{Int64}}\nvertices_by_tree::Vector{Vector{Int64}}\nreverse_bfs_orders::Vector{Vector{Tuple{Int64, Int64}}}\nbuffer::Vector\n\nSee also\n\nAbstractColoringResult\n\n\n\n\n\n","category":"type"},{"location":"dev/#SparseMatrixColorings.LinearSystemColoringResult","page":"Dev docs","title":"SparseMatrixColorings.LinearSystemColoringResult","text":"struct LinearSystemColoringResult{M, R, F} <: AbstractColoringResult{:symmetric, :column, :substitution, M}\n\nStorage for the result of a symmetric coloring with any decompression.\n\nFields\n\nSee the docstring of ColumnColoringResult.\n\nS::Any\ncolor::Vector{Int64}\ngroup::Vector{Vector{Int64}}\nstrict_upper_nonzero_inds::Vector{Tuple{Int64, Int64}}\nstrict_upper_nonzeros_A::Vector\nT_factorization::Any\n\nSee also\n\nAbstractColoringResult\n\n\n\n\n\n","category":"type"},{"location":"dev/#Testing","page":"Dev docs","title":"Testing","text":"","category":"section"},{"location":"dev/","page":"Dev docs","title":"Dev docs","text":"SparseMatrixColorings.directly_recoverable_columns\nSparseMatrixColorings.symmetrically_orthogonal_columns\nSparseMatrixColorings.structurally_orthogonal_columns","category":"page"},{"location":"dev/#SparseMatrixColorings.directly_recoverable_columns","page":"Dev docs","title":"SparseMatrixColorings.directly_recoverable_columns","text":"directly_recoverable_columns(\n    A::AbstractMatrix, color::AbstractVector{<:Integer}\n    verbose=false\n)\n\nReturn true if coloring the columns of the symmetric matrix A with the vector color results in a column-compressed representation that preserves every unique value, thus making direct recovery possible.\n\nwarning: Warning\nThis function is not coded with efficiency in mind, it is designed for small-scale tests.\n\nReferences\n\nWhat Color Is Your Jacobian? Graph Coloring for Computing Derivatives, Gebremedhin et al. (2005)\n\n\n\n\n\n","category":"function"},{"location":"dev/#SparseMatrixColorings.symmetrically_orthogonal_columns","page":"Dev docs","title":"SparseMatrixColorings.symmetrically_orthogonal_columns","text":"symmetrically_orthogonal_columns(\n    A::AbstractMatrix, color::AbstractVector{<:Integer};\n    verbose=false\n)\n\nReturn true if coloring the columns of the symmetric matrix A with the vector color results in a partition that is symmetrically orthogonal, and false otherwise.\n\nA partition of the columns of a symmetrix matrix A is symmetrically orthogonal if, for every nonzero element A[i, j], either of the following statements holds:\n\nthe group containing the column A[:, j] has no other column with a nonzero in row i\nthe group containing the column A[:, i] has no other column with a nonzero in row j\n\nwarning: Warning\nThis function is not coded with efficiency in mind, it is designed for small-scale tests.\n\nReferences\n\nWhat Color Is Your Jacobian? Graph Coloring for Computing Derivatives, Gebremedhin et al. (2005)\n\n\n\n\n\n","category":"function"},{"location":"dev/#SparseMatrixColorings.structurally_orthogonal_columns","page":"Dev docs","title":"SparseMatrixColorings.structurally_orthogonal_columns","text":"structurally_orthogonal_columns(\n    A::AbstractMatrix, color::AbstractVector{<:Integer}\n    verbose=false\n)\n\nReturn true if coloring the columns of the matrix A with the vector color results in a partition that is structurally orthogonal, and false otherwise.\n\nA partition of the columns of a matrix A is structurally orthogonal if, for every nonzero element A[i, j], the group containing column A[:, j] has no other column with a nonzero in row i.\n\nwarning: Warning\nThis function is not coded with efficiency in mind, it is designed for small-scale tests.\n\nReferences\n\nWhat Color Is Your Jacobian? Graph Coloring for Computing Derivatives, Gebremedhin et al. (2005)\n\n\n\n\n\n","category":"function"},{"location":"dev/#Matrix-handling","page":"Dev docs","title":"Matrix handling","text":"","category":"section"},{"location":"dev/","page":"Dev docs","title":"Dev docs","text":"SparseMatrixColorings.respectful_similar\nSparseMatrixColorings.matrix_versions\nSparseMatrixColorings.same_pattern","category":"page"},{"location":"dev/#SparseMatrixColorings.respectful_similar","page":"Dev docs","title":"SparseMatrixColorings.respectful_similar","text":"respectful_similar(A::AbstractMatrix)\nrespectful_similar(A::AbstractMatrix, ::Type{T})\n\nLike Base.similar but returns a transpose or adjoint when A is a transpose or adjoint.\n\n\n\n\n\n","category":"function"},{"location":"dev/#SparseMatrixColorings.matrix_versions","page":"Dev docs","title":"SparseMatrixColorings.matrix_versions","text":"matrix_versions(A::AbstractMatrix)\n\nReturn various versions of the same matrix:\n\ndense and sparse\ntranspose and adjoint\n\nUsed for internal testing.\n\n\n\n\n\n","category":"function"},{"location":"dev/#SparseMatrixColorings.same_pattern","page":"Dev docs","title":"SparseMatrixColorings.same_pattern","text":"same_pattern(A::AbstractMatrix, B::AbstractMatrix)\n\nPerform a partial equality check on the sparsity patterns of A and B:\n\nif the return is true, they might have the same sparsity pattern but we're not sure\nif the return is false, they definitely don't have the same sparsity pattern\n\n\n\n\n\n","category":"function"},{"location":"dev/#Examples","page":"Dev docs","title":"Examples","text":"","category":"section"},{"location":"dev/","page":"Dev docs","title":"Dev docs","text":"SparseMatrixColorings.Example\nSparseMatrixColorings.what_fig_41\nSparseMatrixColorings.what_fig_61\nSparseMatrixColorings.efficient_fig_1\nSparseMatrixColorings.efficient_fig_4","category":"page"},{"location":"dev/#SparseMatrixColorings.Example","page":"Dev docs","title":"SparseMatrixColorings.Example","text":"struct Example{TA<:(AbstractMatrix), TB<:(AbstractMatrix)}\n\nExample coloring problem from one of our reference articles.\n\nUsed for internal testing.\n\nFields\n\nA::AbstractMatrix: decompressed matrix\nB::AbstractMatrix: column-compressed matrix\ncolor::Vector{Int64}: vector of colors\n\n\n\n\n\n","category":"type"},{"location":"dev/#SparseMatrixColorings.what_fig_41","page":"Dev docs","title":"SparseMatrixColorings.what_fig_41","text":"what_fig_41()\n\nConstruct an Example from Figure 4.1 of \"What color is your Jacobian?\", where the nonzero entries are filled with unique values.\n\n\n\n\n\n","category":"function"},{"location":"dev/#SparseMatrixColorings.what_fig_61","page":"Dev docs","title":"SparseMatrixColorings.what_fig_61","text":"what_fig_61()\n\nConstruct an Example from Figure 6.1 of \"What color is your Jacobian?\", where the nonzero entries are filled with unique values.\n\n\n\n\n\n","category":"function"},{"location":"dev/#SparseMatrixColorings.efficient_fig_1","page":"Dev docs","title":"SparseMatrixColorings.efficient_fig_1","text":"efficient_fig_1()\n\nConstruct an Example from Figure 1 of \"Efficient computation of sparse hessians using coloring and AD\", where the nonzero entries are filled with unique values.\n\n\n\n\n\n","category":"function"},{"location":"dev/#SparseMatrixColorings.efficient_fig_4","page":"Dev docs","title":"SparseMatrixColorings.efficient_fig_4","text":"efficient_fig_4()\n\nConstruct an Example from Figure 4 of \"Efficient computation of sparse hessians using coloring and AD\", where the nonzero entries are filled with unique values.\n\n\n\n\n\n","category":"function"},{"location":"#SparseMatrixColorings.jl","page":"Home","title":"SparseMatrixColorings.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Build Status) (Image: Stable Documentation) (Image: Dev Documentation) (Image: Coverage) (Image: Code Style: Blue) (Image: DOI)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Coloring algorithms for sparse Jacobian and Hessian matrices.","category":"page"},{"location":"#Getting-started","page":"Home","title":"Getting started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To install this package, run the following in a Julia Pkg REPL:","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add SparseMatrixColorings","category":"page"},{"location":"#Background","page":"Home","title":"Background","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The algorithms implemented in this package are taken from the following articles:","category":"page"},{"location":"","page":"Home","title":"Home","text":"What Color Is Your Jacobian? Graph Coloring for Computing Derivatives, Gebremedhin et al. (2005)\nNew Acyclic and Star Coloring Algorithms with Application to Computing Hessians, Gebremedhin et al. (2007)\nEfficient Computation of Sparse Hessians Using Coloring and Automatic Differentiation, Gebremedhin et al. (2009)\nColPack: Software for graph coloring and related problems in scientific computing, Gebremedhin et al. (2013)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Some parts of the articles (like definitions) are thus copied verbatim in the documentation.","category":"page"},{"location":"#Alternatives","page":"Home","title":"Alternatives","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"ColPack.jl: a Julia interface to the C++ library ColPack\nSparseDiffTools.jl: contains Julia implementations of some coloring algorithms","category":"page"},{"location":"#Citing","page":"Home","title":"Citing","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Please cite this software using the Zenodo DOI of the version you used. The link https://zenodo.org/doi/10.5281/zenodo.11314275 resolves to the latest version.","category":"page"}]
}
