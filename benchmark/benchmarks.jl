using BenchmarkTools
using LinearAlgebra
using SparseMatrixColorings
using SparseArrays

SUITE = BenchmarkGroup()

for structure in [:nonsymmetric, :symmetric],
    partition in (structure == :nonsymmetric ? [:column, :row] : [:column]),
    decompression in (structure == :nonsymmetric ? [:direct] : [:direct, :substitution]),
    n in [10^3],
    p in [2 / n, 5 / n]

    problem = ColoringProblem(; structure, partition)
    algo = GreedyColoringAlgorithm(; decompression)

    SUITE[:coloring][structure][partition][decompression]["n=$n"]["p=$p"] = @benchmarkable coloring(
        A, $problem, $algo
    ) setup = ( #
        A = sparse(Symmetric(sprand($n, $n, $p)))
    )

    SUITE[:decompress][structure][partition][decompression]["n=$n"]["p=$p"] = @benchmarkable decompress(
        B, result
    ) setup = ( #
        A = sparse(Symmetric(sprand($n, $n, $p)));
        result = coloring(A, $problem, $algo);
        B = compress(A, result)
    )
end
