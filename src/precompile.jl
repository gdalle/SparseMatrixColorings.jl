for (structure, partition, decompression) in [
    (:nonsymmetric, :column, :direct),
    (:nonsymmetric, :row, :direct),
    (:symmetric, :column, :direct),
    (:symmetric, :column, :substitution),
    (:nonsymmetric, :bidirectional, :direct),
    (:nonsymmetric, :bidirectional, :substitution),
]
    A = sparse(Symmetric(sprand(Bool, 100, 100, 0.1)))
    problem = ColoringProblem(; structure, partition)
    algo = GreedyColoringAlgorithm(; decompression, postprocessing=true)
    result = coloring(A, problem, algo)
    if partition == :bidirectional
        Br, Bc = compress(A, result)
        decompress(Br, Bc, result)
    else
        B = compress(A, result)
        decompress(B, result)
    end
end
