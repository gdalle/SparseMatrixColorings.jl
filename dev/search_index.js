var documenterSearchIndex = {"docs":
[{"location":"api/#API-reference","page":"API reference","title":"API reference","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"CollapsedDocStrings = true\nCurrentModule = SparseMatrixColorings","category":"page"},{"location":"api/#Public,-exported","page":"API reference","title":"Public, exported","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"SparseMatrixColorings","category":"page"},{"location":"api/#SparseMatrixColorings.SparseMatrixColorings","page":"API reference","title":"SparseMatrixColorings.SparseMatrixColorings","text":"SparseMatrixColorings\n\nSparseMatrixColorings.jl\n\n(Image: Build Status) (Image: Stable Documentation) (Image: Dev Documentation) (Image: Coverage) (Image: Code Style: Blue)\n\nColoring algorithms for sparse Jacobian and Hessian matrices.\n\nGetting started\n\nTo install this package, run the following in a Julia Pkg REPL:\n\npkg> add SparseMatrixColorings\n\nBackground\n\nThe algorithms implemented in this package are taken from the following articles:\n\nWhat Color Is Your Jacobian? Graph Coloring for Computing Derivatives, Gebremedhin et al. (2005)\nNew Acyclic and Star Coloring Algorithms with Application to Computing Hessians, Gebremedhin et al. (2007)\nEfficient Computation of Sparse Hessians Using Coloring and Automatic Differentiation, Gebremedhin et al. (2009)\nColPack: Software for graph coloring and related problems in scientific computing, Gebremedhin et al. (2013)\n\nSome parts of the articles (like definitions) are thus copied verbatim in the documentation.\n\nAlternatives\n\nColPack.jl: a Julia interface to the C++ library ColPack\nSparseDiffTools.jl: contains Julia implementations of some coloring algorithms\n\nExports\n\nAbstractColoringResult\nColoringProblem\nGreedyColoringAlgorithm\ncoloring\ncolumn_coloring\ncolumn_colors\ncolumn_groups\nrow_coloring\nrow_colors\nrow_groups\nsymmetric_coloring\n\n\n\n\n\n","category":"module"},{"location":"api/#Entry-point","page":"API reference","title":"Entry point","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"coloring\nColoringProblem\nGreedyColoringAlgorithm","category":"page"},{"location":"api/#SparseMatrixColorings.coloring","page":"API reference","title":"SparseMatrixColorings.coloring","text":"coloring(\n    S::AbstractMatrix,\n    problem::ColoringProblem,\n    algo::GreedyColoringAlgorithm\n)\n\nSolve a ColoringProblem on the matrix S with a GreedyColoringAlgorithm and return an AbstractColoringResult.\n\nExample\n\n```jldoctest julia> using SparseMatrixColorings, SparseArrays\n\njulia> S = sparse([            0 0 1 1 0 1            1 0 0 0 1 0            0 1 0 0 1 0            0 1 1 0 0 0        ]);\n\njulia> problem = ColoringProblem(structure=:nonsymmetric, partition=:column);\n\njulia> algo = GreedyColoringAlgorithm();\n\njulia> result = coloring(S, problem, algo);\n\njulia> column_colors(result) 6-element Vector{Int64}:  1  1  2  1  2  3\n\njulia> column_groups(result) 3-element Vector{Vector{Int64}}:  [1, 2, 4]  [3, 5]  [6]\n\nSee also\n\nColoringProblem\nGreedyColoringAlgorithm\nAbstractColoringResult\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseMatrixColorings.ColoringProblem","page":"API reference","title":"SparseMatrixColorings.ColoringProblem","text":"ColoringProblem{structure,partition,decompression}\n\nSelector type for the coloring problem to solve, enabling multiple dispatch.\n\nIt is used inside the main function coloring.\n\nConstructor\n\nColoringProblem(;\n    structure::Symbol=:nonsymmetric,\n    partition::Symbol=:column,\n    decompression::Symbol=:direct,\n)\n\nType parameters\n\nstructure::Symbol: either :nonsymmetric or :symmetric\npartition::Symbol: either :column, :row or :bidirectional\ndecompression::Symbol: either :direct or :substitution\n\nLink to automatic differentiation\n\nMatrix coloring is often used in automatic differentiation, and here is the translation guide:\n\nmatrix mode structure partition\nJacobian forward :nonsymmetric :column\nJacobian reverse :nonsymmetric :row\nHessian forward-over-reverse :symmetric :column\n\n\n\n\n\n","category":"type"},{"location":"api/#SparseMatrixColorings.GreedyColoringAlgorithm","page":"API reference","title":"SparseMatrixColorings.GreedyColoringAlgorithm","text":"GreedyColoringAlgorithm <: ADTypes.AbstractColoringAlgorithm\n\nGreedy coloring algorithm for sparse matrices which colors columns or rows one after the other, following a configurable order.\n\nIt is used inside the main function coloring.\n\nConstructor\n\nGreedyColoringAlgorithm(order::AbstractOrder=NaturalOrder())\n\nThe choice of AbstractOrder impacts the resulting number of colors. It defaults to NaturalOrder for reproducibility, but LargestFirst can sometimes be a better option.\n\nADTypes coloring interface\n\nGreedyColoringAlgorithm is a subtype of ADTypes.AbstractColoringAlgorithm, which means the following methods are also applicable:\n\nADTypes.column_coloring\nADTypes.row_coloring\nADTypes.symmetric_coloring\n\n\n\n\n\n","category":"type"},{"location":"api/#Result-analysis","page":"API reference","title":"Result analysis","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"AbstractColoringResult\ncolumn_colors\nrow_colors\ncolumn_groups\nrow_groups","category":"page"},{"location":"api/#SparseMatrixColorings.AbstractColoringResult","page":"API reference","title":"SparseMatrixColorings.AbstractColoringResult","text":"AbstractColoringResult\n\nAbstract type for the result of a coloring algorithm.\n\nIt is the supertype of the object returned by the main function coloring.\n\nApplicable methods\n\ncolumn_colors and column_groups (for a :column or :bidirectional partition) \nrow_colors and row_groups (for a :row or :bidirectional partition)\ndecompress and decompress!\n\nwarning: Warning\nUnlike the methods above, the concrete subtypes of AbstractColoringResult are not part of the public API and may change without notice.\n\n\n\n\n\n","category":"type"},{"location":"api/#SparseMatrixColorings.column_colors","page":"API reference","title":"SparseMatrixColorings.column_colors","text":"column_colors(result::AbstractColoringResult)\n\nReturn a vector color of integer colors, one for each column of the colored matrix.\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseMatrixColorings.row_colors","page":"API reference","title":"SparseMatrixColorings.row_colors","text":"row_colors(result::AbstractColoringResult)\n\nReturn a vector color of integer colors, one for each row of the colored matrix.\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseMatrixColorings.column_groups","page":"API reference","title":"SparseMatrixColorings.column_groups","text":"column_groups(result::AbstractColoringResult)\n\nReturn a vector group such that for every color c, group[c] contains the indices of all columns that are colored with c.\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseMatrixColorings.row_groups","page":"API reference","title":"SparseMatrixColorings.row_groups","text":"row_groups(result::AbstractColoringResult)\n\nReturn a vector group such that for every color c, group[c] contains the indices of all rows that are colored with c.\n\n\n\n\n\n","category":"function"},{"location":"api/#Public,-not-exported","page":"API reference","title":"Public, not exported","text":"","category":"section"},{"location":"api/#Decompression","page":"API reference","title":"Decompression","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"decompress\ndecompress!","category":"page"},{"location":"api/#SparseMatrixColorings.decompress","page":"API reference","title":"SparseMatrixColorings.decompress","text":"decompress(B::AbstractMatrix, result::AbstractColoringResult)\n\nDecompress B out-of-place into a new matrix A, given a coloring result of the sparsity pattern of A.\n\nSee also\n\nAbstractColoringResult\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseMatrixColorings.decompress!","page":"API reference","title":"SparseMatrixColorings.decompress!","text":"decompress!(\n    A::AbstractMatrix, B::AbstractMatrix,\n    result::AbstractColoringResult,\n)\n\nDecompress B in-place into an existing matrix A, given a coloring result of the sparsity pattern of A.\n\nSee also\n\nAbstractColoringResult\n\n\n\n\n\n","category":"function"},{"location":"api/#Orders","page":"API reference","title":"Orders","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"AbstractOrder\nNaturalOrder\nRandomOrder\nLargestFirst","category":"page"},{"location":"api/#SparseMatrixColorings.AbstractOrder","page":"API reference","title":"SparseMatrixColorings.AbstractOrder","text":"AbstractOrder\n\nAbstract supertype for the vertex order used inside GreedyColoringAlgorithm.\n\nIn this algorithm, the rows and columns of a matrix form a graph, and the vertices are colored one after the other in a greedy fashion. Depending on how the vertices are ordered, the number of colors necessary may vary.\n\nSubtypes\n\nNaturalOrder\nRandomOrder\nLargestFirst\n\n\n\n\n\n","category":"type"},{"location":"api/#SparseMatrixColorings.NaturalOrder","page":"API reference","title":"SparseMatrixColorings.NaturalOrder","text":"NaturalOrder()\n\nInstance of AbstractOrder which sorts vertices using their index in the provided graph.\n\n\n\n\n\n","category":"type"},{"location":"api/#SparseMatrixColorings.RandomOrder","page":"API reference","title":"SparseMatrixColorings.RandomOrder","text":"RandomOrder(rng=default_rng())\n\nInstance of AbstractOrder which sorts vertices using a random permutation.\n\n\n\n\n\n","category":"type"},{"location":"api/#SparseMatrixColorings.LargestFirst","page":"API reference","title":"SparseMatrixColorings.LargestFirst","text":"LargestFirst()\n\nInstance of AbstractOrder which sorts vertices using their degree in the provided graph: the largest degree comes first.\n\n\n\n\n\n","category":"type"},{"location":"api/#Private","page":"API reference","title":"Private","text":"","category":"section"},{"location":"api/#Graph-storage","page":"API reference","title":"Graph storage","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"SparseMatrixColorings.Graph\nSparseMatrixColorings.BipartiteGraph\nSparseMatrixColorings.vertices\nSparseMatrixColorings.neighbors\nSparseMatrixColorings.adjacency_graph\nSparseMatrixColorings.bipartite_graph","category":"page"},{"location":"api/#SparseMatrixColorings.Graph","page":"API reference","title":"SparseMatrixColorings.Graph","text":"Graph{T}\n\nUndirected graph structure stored in Compressed Sparse Column (CSC) format.\n\nFields\n\ncolptr::Vector{T}: same as for SparseMatrixCSC\nrowval::Vector{T}: same as for SparseMatrixCSC\n\n\n\n\n\n","category":"type"},{"location":"api/#SparseMatrixColorings.BipartiteGraph","page":"API reference","title":"SparseMatrixColorings.BipartiteGraph","text":"BipartiteGraph{T}\n\nUndirected bipartite graph structure stored in bidirectional Compressed Sparse Column format (redundancy allows for faster access).\n\nA bipartite graph has two \"sides\", which we number 1 and 2.\n\nFields\n\ng1::Graph{T}: contains the neighbors for vertices on side 1\ng2::Graph{T}: contains the neighbors for vertices on side 2\n\n\n\n\n\n","category":"type"},{"location":"api/#SparseMatrixColorings.vertices","page":"API reference","title":"SparseMatrixColorings.vertices","text":"vertices(bg::BipartiteGraph, Val(side))\n\nReturn the list of vertices of bg from the specified side as a range 1:n.\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseMatrixColorings.neighbors","page":"API reference","title":"SparseMatrixColorings.neighbors","text":"neighbors(bg::BipartiteGraph, Val(side), v::Integer)\n\nReturn the neighbors of v (a vertex from the specified side, 1 or 2), in the graph bg.\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseMatrixColorings.adjacency_graph","page":"API reference","title":"SparseMatrixColorings.adjacency_graph","text":"adjacency_graph(H::AbstractMatrix)\n\nReturn a Graph representing the nonzeros of a symmetric matrix (typically a Hessian matrix).\n\nThe adjacency graph of a symmetrix matric A ∈ ℝ^{n × n} is G(A) = (V, E) where\n\nV = 1:n is the set of rows or columns i/j\n(i, j) ∈ E whenever A[i, j] ≠ 0 and i ≠ j\n\nReferences\n\nWhat Color Is Your Jacobian? Graph Coloring for Computing Derivatives, Gebremedhin et al. (2005)\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseMatrixColorings.bipartite_graph","page":"API reference","title":"SparseMatrixColorings.bipartite_graph","text":"bipartite_graph(J::AbstractMatrix)\n\nReturn a BipartiteGraph representing the nonzeros of a non-symmetric matrix (typically a Jacobian matrix).\n\nThe bipartite graph of a matrix A ∈ ℝ^{m × n} is Gb(A) = (V₁, V₂, E) where\n\nV₁ = 1:m is the set of rows i\nV₂ = 1:n is the set of columns j\n(i, j) ∈ E whenever A[i, j] ≠ 0\n\nReferences\n\nWhat Color Is Your Jacobian? Graph Coloring for Computing Derivatives, Gebremedhin et al. (2005)\n\n\n\n\n\n","category":"function"},{"location":"api/#Low-level-coloring","page":"API reference","title":"Low-level coloring","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"SparseMatrixColorings.partial_distance2_coloring\nSparseMatrixColorings.symmetric_coefficient\nSparseMatrixColorings.star_coloring\nSparseMatrixColorings.StarSet\nSparseMatrixColorings.group_by_color\nSparseMatrixColorings.get_matrix","category":"page"},{"location":"api/#SparseMatrixColorings.partial_distance2_coloring","page":"API reference","title":"SparseMatrixColorings.partial_distance2_coloring","text":"partial_distance2_coloring(bg::BipartiteGraph, ::Val{side}, order::AbstractOrder)\n\nCompute a distance-2 coloring of the given side (1 or 2) in the bipartite graph bg and return a vector of integer colors.\n\nA distance-2 coloring is such that two vertices have different colors if they are at distance at most 2.\n\nThe vertices are colored in a greedy fashion, following the order supplied.\n\nSee also\n\nBipartiteGraph\nAbstractOrder\n\nReferences\n\nWhat Color Is Your Jacobian? Graph Coloring for Computing Derivatives, Gebremedhin et al. (2005), Algorithm 3.2\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseMatrixColorings.symmetric_coefficient","page":"API reference","title":"SparseMatrixColorings.symmetric_coefficient","text":"symmetric_coefficient(\n    i::Integer, j::Integer,\n    color::AbstractVector{<:Integer},\n    group::AbstractVector{<:AbstractVector{<:Integer}},\n    S::AbstractMatrix{Bool}\n)\n\nsymmetric_coefficient(\n    i::Integer, j::Integer,\n    color::AbstractVector{<:Integer},\n    star_set::StarSet\n)\n\nReturn the indices (k, c) such that A[i, j] = B[k, c], where A is the uncompressed symmetric matrix and B is the column-compressed matrix.\n\nThe first version corresponds to algorithm DirectRecover1 in the paper, the second to DirectRecover2.\n\nReferences\n\nEfficient Computation of Sparse Hessians Using Coloring and Automatic Differentiation, Gebremedhin et al. (2009), Figures 2 and 3\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseMatrixColorings.star_coloring","page":"API reference","title":"SparseMatrixColorings.star_coloring","text":"star_coloring(g::Graph, order::AbstractOrder)\n\nCompute a star coloring of all vertices in the adjacency graph g and return a tuple (color, star_set), where\n\ncolor is the vector of integer colors\nstar_set is a StarSet encoding the set of 2-colored stars\n\nA star coloring is a distance-1 coloring such that every path on 4 vertices uses at least 3 colors.\n\nThe vertices are colored in a greedy fashion, following the order supplied.\n\nSee also\n\nGraph\nAbstractOrder\n\nReferences\n\nNew Acyclic and Star Coloring Algorithms with Application to Computing Hessians, Gebremedhin et al. (2007), Algorithm 4.1\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseMatrixColorings.StarSet","page":"API reference","title":"SparseMatrixColorings.StarSet","text":"struct StarSet\n\nEncode a set of 2-colored stars resulting from the star coloring algorithm.\n\nFields\n\nstar::Dict{Tuple{Int64, Int64}, Int64}: a mapping from edges (pair of vertices) their to star index\nhub::Vector{Int64}: a mapping from star indices to their hub (the hub is 0 if the star only contains one edge)\n\nReferences\n\nNew Acyclic and Star Coloring Algorithms with Application to Computing Hessians, Gebremedhin et al. (2007), Algorithm 4.1\n\n\n\n\n\n","category":"type"},{"location":"api/#SparseMatrixColorings.group_by_color","page":"API reference","title":"SparseMatrixColorings.group_by_color","text":"group_by_color(color::Vector{Int})\n\nCreate group::Vector{Vector{Int}} such that i ∈ group[c] iff color[i] == c.\n\nAssumes the colors are contiguously numbered from 1 to some cmax.\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseMatrixColorings.get_matrix","page":"API reference","title":"SparseMatrixColorings.get_matrix","text":"get_matrix(result::AbstractColoringResult)\n\nReturn the matrix that was colored.\n\n\n\n\n\n","category":"function"},{"location":"api/#Concrete-coloring-results","page":"API reference","title":"Concrete coloring results","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"SparseMatrixColorings.DefaultColoringResult\nSparseMatrixColorings.DirectSparseColoringResult","category":"page"},{"location":"api/#SparseMatrixColorings.DefaultColoringResult","page":"API reference","title":"SparseMatrixColorings.DefaultColoringResult","text":"struct DefaultColoringResult{structure, partition, decompression, M} <: AbstractColoringResult{structure, partition, decompression, M}\n\nDefault storage for the result of a coloring algorithm, containing minimal information.\n\nFields\n\nmatrix::Any: matrix that was colored\ncolor::Vector{Int64}: one integer color for each column or row (depending on partition)\ngroup::Vector{Vector{Int64}}: color groups for columns or rows (depending on partition)\n\nSee also\n\nAbstractColoringResult\n\n\n\n\n\n","category":"type"},{"location":"api/#SparseMatrixColorings.DirectSparseColoringResult","page":"API reference","title":"SparseMatrixColorings.DirectSparseColoringResult","text":"struct DirectSparseColoringResult{structure, partition, M} <: AbstractColoringResult{structure, partition, :direct, M}\n\nStorage for the result of a coloring algorithm when the decompression target is a SparseMatrixCSC.\n\nFields\n\nmatrix::Any: matrix that was colored\ncolor::Vector{Int64}: one integer color for each column or row (depending on partition)\ngroup::Vector{Vector{Int64}}: color groups for columns or rows (depending on partition)\ncompressed_indices::Vector{Int64}: flattened indices mapping the compressed matrix B to the uncompressed matrix A: they satisfy nonzeros(A)[k] = vec(B)[compressed_indices[k]]\n\nSee also\n\nAbstractColoringResult\n\n\n\n\n\n","category":"type"},{"location":"api/#Testing","page":"API reference","title":"Testing","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"SparseMatrixColorings.same_sparsity_pattern\nSparseMatrixColorings.directly_recoverable_columns\nSparseMatrixColorings.symmetrically_orthogonal_columns\nSparseMatrixColorings.structurally_orthogonal_columns","category":"page"},{"location":"api/#SparseMatrixColorings.same_sparsity_pattern","page":"API reference","title":"SparseMatrixColorings.same_sparsity_pattern","text":"same_sparsity_pattern(A::AbstractMatrix, B::AbstractMatrix)\n\nPerform a partial equality check on the sparsity patterns of A and B:\n\nif the return is true, they might have the same sparsity pattern but we're not sure\nif the return is false, they definitely don't have the same sparsity pattern\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseMatrixColorings.directly_recoverable_columns","page":"API reference","title":"SparseMatrixColorings.directly_recoverable_columns","text":"directly_recoverable_columns(\n    A::AbstractMatrix, color::AbstractVector{<:Integer}\n    verbose=false\n)\n\nReturn true if coloring the columns of the symmetric matrix A with the vector color results in a column-compressed representation that preserves every unique value, thus making direct recovery possible.\n\nwarning: Warning\nThis function is not coded with efficiency in mind, it is designed for small-scale tests.\n\nReferences\n\nWhat Color Is Your Jacobian? Graph Coloring for Computing Derivatives, Gebremedhin et al. (2005)\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseMatrixColorings.symmetrically_orthogonal_columns","page":"API reference","title":"SparseMatrixColorings.symmetrically_orthogonal_columns","text":"symmetrically_orthogonal_columns(\n    A::AbstractMatrix, color::AbstractVector{<:Integer};\n    verbose=false\n)\n\nReturn true if coloring the columns of the symmetric matrix A with the vector color results in a partition that is symmetrically orthogonal, and false otherwise.\n\nA partition of the columns of a symmetrix matrix A is symmetrically orthogonal if, for every nonzero element A[i, j], either of the following statements holds:\n\nthe group containing the column A[:, j] has no other column with a nonzero in row i\nthe group containing the column A[:, i] has no other column with a nonzero in row j\n\nwarning: Warning\nThis function is not coded with efficiency in mind, it is designed for small-scale tests.\n\nReferences\n\nWhat Color Is Your Jacobian? Graph Coloring for Computing Derivatives, Gebremedhin et al. (2005)\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseMatrixColorings.structurally_orthogonal_columns","page":"API reference","title":"SparseMatrixColorings.structurally_orthogonal_columns","text":"structurally_orthogonal_columns(\n    A::AbstractMatrix, color::AbstractVector{<:Integer}\n    verbose=false\n)\n\nReturn true if coloring the columns of the matrix A with the vector color results in a partition that is structurally orthogonal, and false otherwise.\n\nA partition of the columns of a matrix A is structurally orthogonal if, for every nonzero element A[i, j], the group containing column A[:, j] has no other column with a nonzero in row i.\n\nwarning: Warning\nThis function is not coded with efficiency in mind, it is designed for small-scale tests.\n\nReferences\n\nWhat Color Is Your Jacobian? Graph Coloring for Computing Derivatives, Gebremedhin et al. (2005)\n\n\n\n\n\n","category":"function"},{"location":"api/#Matrix-handling","page":"API reference","title":"Matrix handling","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"SparseMatrixColorings.respectful_similar\nSparseMatrixColorings.matrix_versions","category":"page"},{"location":"api/#SparseMatrixColorings.respectful_similar","page":"API reference","title":"SparseMatrixColorings.respectful_similar","text":"respectful_similar(A::AbstractMatrix)\nrespectful_similar(A::AbstractMatrix, ::Type{T})\n\nLike Base.similar but returns a transpose or adjoint when A is a transpose or adjoint.\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseMatrixColorings.matrix_versions","page":"API reference","title":"SparseMatrixColorings.matrix_versions","text":"matrix_versions(A::AbstractMatrix)\n\nReturn various versions of the same matrix:\n\ndense and sparse\ntranspose and adjoint\n\nUsed for internal testing.\n\n\n\n\n\n","category":"function"},{"location":"api/#Examples","page":"API reference","title":"Examples","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"SparseMatrixColorings.Example\nSparseMatrixColorings.what_fig_41\nSparseMatrixColorings.efficient_fig_1","category":"page"},{"location":"api/#SparseMatrixColorings.Example","page":"API reference","title":"SparseMatrixColorings.Example","text":"struct Example{TA<:(AbstractMatrix), TB<:(AbstractMatrix)}\n\nExample coloring problem from one of our reference articles.\n\nUsed for internal testing.\n\nFields\n\nA::AbstractMatrix: decompressed matrix\nB::AbstractMatrix: column-compressed matrix\ncolor::Vector{Int64}: vector of colors\n\n\n\n\n\n","category":"type"},{"location":"api/#SparseMatrixColorings.what_fig_41","page":"API reference","title":"SparseMatrixColorings.what_fig_41","text":"what_fig_41()\n\nConstruct an Example from Figure 4.1 of \"What color is your Jacobian?\", where the nonzero entries are filled with unique values.\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseMatrixColorings.efficient_fig_1","page":"API reference","title":"SparseMatrixColorings.efficient_fig_1","text":"efficient_fig_1()\n\nConstruct an Example from Figure 1 of \"Efficient computation of sparse hessians using coloring and AD\", where the nonzero entries are filled with unique values.\n\n\n\n\n\n","category":"function"},{"location":"#SparseMatrixColorings.jl","page":"Home","title":"SparseMatrixColorings.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Build Status) (Image: Stable Documentation) (Image: Dev Documentation) (Image: Coverage) (Image: Code Style: Blue)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Coloring algorithms for sparse Jacobian and Hessian matrices.","category":"page"},{"location":"#Getting-started","page":"Home","title":"Getting started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To install this package, run the following in a Julia Pkg REPL:","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add SparseMatrixColorings","category":"page"},{"location":"#Background","page":"Home","title":"Background","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The algorithms implemented in this package are taken from the following articles:","category":"page"},{"location":"","page":"Home","title":"Home","text":"What Color Is Your Jacobian? Graph Coloring for Computing Derivatives, Gebremedhin et al. (2005)\nNew Acyclic and Star Coloring Algorithms with Application to Computing Hessians, Gebremedhin et al. (2007)\nEfficient Computation of Sparse Hessians Using Coloring and Automatic Differentiation, Gebremedhin et al. (2009)\nColPack: Software for graph coloring and related problems in scientific computing, Gebremedhin et al. (2013)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Some parts of the articles (like definitions) are thus copied verbatim in the documentation.","category":"page"},{"location":"#Alternatives","page":"Home","title":"Alternatives","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"ColPack.jl: a Julia interface to the C++ library ColPack\nSparseDiffTools.jl: contains Julia implementations of some coloring algorithms","category":"page"}]
}