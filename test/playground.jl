using SparseArrays, LinearAlgebra
using SparseMatrixColorings

algo = GreedyColoringAlgorithm(; decompression=:direct)
problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)

A = sparse(Symmetric(sprand(1000, 1000, 0.05)))

@profview for _ in 1:1000; coloring(A, problem, algo); end
@profview_allocs for _ in 1:10000; coloring(A, problem, algo); end

using SparseMatrixColorings: group_by_color

using BenchmarkTools
@btime coloring($A, $problem, $algo);
color = rand(1:100, 1000)
group_by_color(color)
