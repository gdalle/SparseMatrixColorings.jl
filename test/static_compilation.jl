# Guard for static compilation: `juliac --trim=safe` rejects any call whose
# result type is not fully inferred.

using SparseArrays
using SparseMatrixColorings
using Test

const STATIC_COMBOS = [
    (ColoringProblem{:nonsymmetric,:column}(), GreedyColoringAlgorithm{:direct}()),
    (ColoringProblem{:nonsymmetric,:row}(), GreedyColoringAlgorithm{:direct}()),
    (ColoringProblem{:symmetric,:column}(), GreedyColoringAlgorithm{:direct}()),
    (ColoringProblem{:symmetric,:column}(), GreedyColoringAlgorithm{:substitution}()),
    (ColoringProblem{:nonsymmetric,:bidirectional}(), GreedyColoringAlgorithm{:direct}()),
    (
        ColoringProblem{:nonsymmetric,:bidirectional}(),
        GreedyColoringAlgorithm{:substitution}(),
    ),
]

# Symmetric, so that it is a valid input for every combination above.
const STATIC_MATRIX = sparse(
    [1, 2, 1, 2, 3, 2, 3, 4, 3, 4], [1, 1, 2, 2, 2, 3, 3, 3, 4, 4], ones(10), 4, 4
)

@testset "coloring infers a concrete result type" begin
    @testset "$(typeof(problem)) / $(typeof(algo))" for (problem, algo) in STATIC_COMBOS
        for decompression_eltype in (Float32, Float64)
            for symmetric_pattern in (false, true)
                @test (@inferred(
                    (
                        (A, p, a, sp, R) ->
                            coloring(A, p, a; decompression_eltype=R, symmetric_pattern=sp)
                    )(
                        STATIC_MATRIX,
                        problem,
                        algo,
                        symmetric_pattern,
                        decompression_eltype,
                    )
                )) isa AbstractColoringResult
            end
        end
    end
end
