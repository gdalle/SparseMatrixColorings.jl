struct ChordalColoringAlgorithm <: ADTypes.AbstractColoringAlgorithm end

struct ChordalColoringResult{M <: AbstractMatrix, V} <: AbstractColoringResult{:symmetric, :column, :direct}
    A::M
    color::Vector{Int}
    group::V
end

function coloring(
    A::AbstractMatrix,
    ::ColoringProblem{:symmetric,:column},
    algo::ChordalColoringAlgorithm,
)
    # compute an elimination ordering using
    # the maximum cardinality search algorithm
    perm, invp = CliqueTrees.permutation(A; alg=CliqueTrees.MCS())
    
    # if the ordering is not perfect, then the graph is not chordal
    if !CliqueTrees.isperfect(A, perm, invp)
        error("Matrix does not have a chordal sparsity pattern.")
    end

    # if the graph is chordal, then find a minimal vertex coloring
    color = CliqueTrees.color(A, perm, invp).colors

    # compute groups and return result
    group = group_by_color(color)
    return ChordalColoringResult(A, color, group)
end
