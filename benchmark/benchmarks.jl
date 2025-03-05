using BenchmarkTools
using LinearAlgebra
using SparseMatrixColorings
using SparseArrays
using StableRNGs

SUITE = BenchmarkGroup()

for structure in [:nonsymmetric, :symmetric],
    partition in (structure == :nonsymmetric ? [:column, :row, :bidirectional] : [:column]),
    decompression in (
        if (structure == :nonsymmetric && partition in [:column, :row])
            [:direct]
        else
            [:direct, :substitution]
        end
    ),
    n in [10^3, 10^5],
    p in [2 / n, 5 / n, 10 / n]

    problem = ColoringProblem(; structure, partition)
    algo = GreedyColoringAlgorithm(
        RandomOrder(StableRNG(0), 0); decompression, postprocessing=true
    )

    # use several random matrices to reduce variance
    nb_samples = 5
    As = [sparse(Symmetric(sprand(StableRNG(i), Bool, n, n, p))) for i in 1:nb_samples]
    results = [coloring(A, problem, algo; decompression_eltype=Float64) for A in As]
    Bs = [compress(Float64.(A), result) for (A, result) in zip(As, results)]

    SUITE[:coloring][structure][partition][decompression]["n=$n"]["p=$p"] = @benchmarkable begin
        for A in $As
            coloring(A, $problem, $algo)
        end
    end

    SUITE[:decompress][structure][partition][decompression]["n=$n"]["p=$p"] = @benchmarkable begin
        for (B, result) in zip($Bs, $results)
            if B isa AbstractMatrix
                decompress(B, result)
            elseif B isa Tuple
                decompress(B[1], B[2], result)
            end
        end
    end
end
